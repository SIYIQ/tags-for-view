import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk # Keep PIL imports
try:
    RESAMPLING_METHOD = Image.Resampling.LANCZOS
except AttributeError:
    try:
        RESAMPLING_METHOD = Image.LANCZOS 
    except AttributeError:
        RESAMPLING_METHOD = Image.ANTIALIAS

from pathlib import Path
import sys
import os
import threading
import queue
import subprocess
from typing import Generator # Removed Iterable as it wasn't used

# --- DB Imports ---
import sqlite3 # Already used by OptimizedImageDB
import hashlib # Already used by OptimizedImageDB
from functools import lru_cache # Already used by OptimizedImageDB
from threading import Lock # Already used by OptimizedImageDB
# --- End DB Imports ---

# Ensure your 'tagger' package is accessible
try:
    from tagger.interrogator.interrogator import AbsInterrogator
    from tagger.interrogators import interrogators
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 'tagger' library: {e}", file=sys.stderr)
    interrogators = {} 
    # No messagebox here, will be handled in TaggerApp.__init__ if needed

# --- OptimizedImageDB Class (Paste the updated class definition here) ---
import sqlite3
import hashlib
from functools import lru_cache
from threading import Lock
from pathlib import Path # Make sure Path is imported

class OptimizedImageDB:
    """优化的图像数据库管理类"""
    def __init__(self, db_path="D:\wd14-tagger-standalone-main\wd14-tagger-standalone-main_0.4\image_tag_cache.db"): # Default to local file
        self.db_lock = Lock() # 添加数据库操作锁
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-200000") # Approx 20MB
        self._create_tables()
        self._create_indexes()
        self._cache = {}
        self._cache_lock = Lock() # Lock for cache operations

    def _create_tables(self):
        with self.db_lock:
            with self.conn:
                self.conn.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    filepath TEXT UNIQUE NOT NULL,
                    file_size INTEGER,
                    file_hash TEXT UNIQUE NOT NULL
                )
                ''')
                self.conn.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
                ''')
                self.conn.execute('''
                CREATE TABLE IF NOT EXISTS image_tags (
                    image_id INTEGER,
                    tag_id INTEGER,
                    PRIMARY KEY(image_id, tag_id),
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
                    FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
                )
                ''')

    def _create_indexes(self):
        with self.db_lock:
            with self.conn:
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_images_filepath ON images(filepath)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_image_tags_image ON image_tags(image_id)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag_id)")

    def bulk_insert(self, images_data):
        """
        批量插入图像及其标签数据。
        images_data: [{'image': (filename, filepath, size, hash), 'tags': [tag1, tag2]}, ...]
        """
        images_to_insert = []
        all_tags_from_batch = set() # All unique tags from this batch
        
        with self.db_lock:
            with self.conn:
                existing_hashes_cursor = self.conn.execute("SELECT file_hash FROM images")
                existing_hashes = set(row[0] for row in existing_hashes_cursor)
        
        # 1. 准备数据
        for data in images_data:
            img_filename, img_filepath, img_size, img_hash = data['image']
            if img_hash and img_hash not in existing_hashes:
                images_to_insert.append((img_filename, str(img_filepath), img_size, img_hash)) # Ensure filepath is string
            # Collect all tags from the batch, regardless of whether the image is new or existing
            for tag in data['tags']:
                all_tags_from_batch.add(tag)
        
        # No new images to insert, but we might still need to insert tags or link existing images to new tags
        if not images_to_insert and not all_tags_from_batch:
            print("没有新的图像或标签需要处理。")
            return

        with self.db_lock:
            with self.conn:
                # 2. 批量插入新图像
                if images_to_insert:
                    self.conn.executemany(
                        '''
                        INSERT OR IGNORE INTO images (filename, filepath, file_size, file_hash)
                        VALUES (?, ?, ?, ?)
                        ''', images_to_insert)

                # 3. 批量插入新标签 (from the entire batch's tags)
                if all_tags_from_batch:
                    tag_tuples = [(tag,) for tag in all_tags_from_batch]
                    self.conn.executemany(
                        '''
                        INSERT OR IGNORE INTO tags (name) VALUES (?)
                        ''', tag_tuples)

                # 4. 获取所有相关图像和标签的 ID
                # Collect ALL unique image hashes and tags from the input `images_data`
                # that we intend to process for linking.
                all_img_hashes_in_images_data = list(set(data['image'][3] for data in images_data if data['image'][3]))
                
                image_id_map = {}
                if all_img_hashes_in_images_data:
                    placeholders = ','.join('?' * len(all_img_hashes_in_images_data))
                    cursor = self.conn.execute(f"SELECT file_hash, id FROM images WHERE file_hash IN ({placeholders})", all_img_hashes_in_images_data)
                    image_id_map = {h: img_id for h, img_id in cursor.fetchall()}

                tag_id_map = {}
                if all_tags_from_batch:
                    placeholders = ','.join('?' * len(all_tags_from_batch))
                    cursor = self.conn.execute(f"SELECT name, id FROM tags WHERE name IN ({placeholders})", list(all_tags_from_batch))
                    tag_id_map = {name: tag_id for name, tag_id in cursor.fetchall()}

                # 5. 准备图像-标签关系数据
                image_tag_relations = []
                for data in images_data:
                    img_hash = data['image'][3]
                    img_id = image_id_map.get(img_hash)
                    
                    if img_id: 
                        for tag_name in data['tags']:
                            tag_id = tag_id_map.get(tag_name)
                            if tag_id: 
                                image_tag_relations.append((img_id, tag_id))
                
                # 6. 批量插入图像-标签关系 (忽略已存在的关系)
                if image_tag_relations:
                    self.conn.executemany(
                        '''
                        INSERT OR IGNORE INTO image_tags (image_id, tag_id) VALUES (?, ?)
                        ''', image_tag_relations)
        # print(f"Bulk insert complete. Processed {len(images_data)} items.")


    def search(self, query_tags: list[str], page=1, page_size=5000, match_all=True):
        """
        根据标签名列表搜索图片文件路径。
        query_tags: 标签名列表, e.g., ["tag1", "cat"]
        page: 页码 (1-indexed)
        page_size: 每页数量
        match_all: True for AND (所有标签都匹配), False for OR (任一标签匹配)
        Returns: (list_of_filepaths, total_matching_items)
        """
        cache_key = f"search_{'_'.join(sorted(query_tags))}_{page}_{page_size}_{match_all}"
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        results = []
        total_items = 0
        
        # Sanitize query_tags: ensure they are strings and handle empty list
        clean_query_tags = [str(t).strip().lower() for t in query_tags if str(t).strip()]
        if not clean_query_tags:
            return [], 0

        with self.db_lock: # Ensure thread safety for DB operations
            with self.conn: # Use a single connection context
                # Build the core query part for selecting image IDs
                # This part is complex for AND logic with multiple tags
                base_query_select = "SELECT i.id FROM images i "
                base_query_where_parts = []
                params = []

                if match_all: # AND logic
                    # Each tag must exist for the image
                    # Using subqueries or JOINs with GROUP BY and COUNT
                    # Example: Images that have ALL specified tags
                    # This is a common way to do "tags AND"
                    placeholders = ','.join('?' for _ in clean_query_tags)
                    sub_query_image_ids = f"""
                        SELECT it.image_id
                        FROM image_tags it
                        JOIN tags t ON it.tag_id = t.id
                        WHERE t.name IN ({placeholders})
                        GROUP BY it.image_id
                        HAVING COUNT(DISTINCT t.id) = ?
                    """
                    params.extend(clean_query_tags)
                    params.append(len(clean_query_tags))
                    
                    # Total count query
                    count_sql = f"SELECT COUNT(DISTINCT image_id) FROM ({sub_query_image_ids}) "
                    count_cursor = self.conn.execute(count_sql, params)
                    total_items = count_cursor.fetchone()[0] if count_cursor else 0

                    # Results query with pagination
                    offset = (page - 1) * page_size
                    # Select filepaths for the image_ids found
                    results_sql = f"""
                        SELECT i.filepath
                        FROM images i
                        WHERE i.id IN ({sub_query_image_ids})
                        ORDER BY i.filepath -- Or i.id, or filename, etc.
                        LIMIT ? OFFSET ?
                    """
                    params.append(page_size)
                    params.append(offset)
                    
                    cursor = self.conn.execute(results_sql, params)
                    results = [Path(row[0]) for row in cursor.fetchall()]

                else: # OR logic (any tag matches using LIKE for fuzzy matching)
                    # This is simpler, but LIKE can be slow on large datasets without FTS
                    # For exact OR matching on tag names:
                    # WHERE t.name IN (?, ?, ?)
                    # For fuzzy OR matching:
                    # WHERE t.name LIKE ? OR t.name LIKE ? ...

                    # We'll do fuzzy OR with LIKE
                    # This means if query_tags is ["cat", "dog"], it searches for "cat" OR "dog"
                    # Each tag in query_tags will be a separate LIKE clause
                    like_clauses = " OR ".join(["t.name LIKE ?"] * len(clean_query_tags))
                    
                    # Total count query
                    count_sql = f"""
                        SELECT COUNT(DISTINCT i.id)
                        FROM images i
                        JOIN image_tags it ON i.id = it.image_id
                        JOIN tags t ON it.tag_id = t.id
                        WHERE {like_clauses}
                    """
                    count_params = [f"%{tag}%" for tag in clean_query_tags]
                    count_cursor = self.conn.execute(count_sql, count_params)
                    total_items = count_cursor.fetchone()[0] if count_cursor else 0
                    
                    # Results query with pagination
                    offset = (page - 1) * page_size
                    results_sql = f"""
                        SELECT DISTINCT i.filepath
                        FROM images i
                        JOIN image_tags it ON i.id = it.image_id
                        JOIN tags t ON it.tag_id = t.id
                        WHERE {like_clauses}
                        ORDER BY i.filepath
                        LIMIT ? OFFSET ?
                    """
                    results_params = count_params + [page_size, offset]
                    cursor = self.conn.execute(results_sql, results_params)
                    results = [Path(row[0]) for row in cursor.fetchall()]
        print(f"DB SEARCH DEBUG: Page: {page}, PageSize: {page_size}, Offset: {offset}")
        print(f"DB SEARCH DEBUG: SQL: {results_sql}")
        print(f"DB SEARCH DEBUG: Params: {results_params if not match_all else params}") # Adjust based on branch
        print(f"DB SEARCH DEBUG: Calculated Total Items: {total_items}")
        # After cursor.execute and fetchall
        print(f"DB SEARCH DEBUG: Fetched {len(results)} results for page {page}")


        with self._cache_lock:
            self._cache[cache_key] = (results, total_items)
            if len(self._cache) > 200: # Cache size limit
                try:
                    del self._cache[next(iter(self._cache))]
                except StopIteration:
                    pass # Cache is empty
        return results, total_items


    def get_all_tags(self):
        with self.db_lock:
            with self.conn:
                cursor = self.conn.execute("SELECT name FROM tags ORDER BY name")
                return [row[0] for row in cursor.fetchall()]

    @lru_cache(maxsize=2048) # Increased cache for file hashes
    def calculate_file_hash(self, file_path_str: str): # Expect string
        hasher = hashlib.md5()
        try:
            with open(file_path_str, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except FileNotFoundError:
            # print(f"警告: 文件未找到，无法计算哈希值: {file_path_str}")
            return None # Return None for UI to handle, less console noise
        except Exception as e:
            print(f"错误: 计算哈希值失败 {file_path_str}: {e}")
            return None

    def get_hash_for_path(self, filepath_str: str): # Expect string
        with self.db_lock:
            with self.conn:
                cursor = self.conn.execute("SELECT file_hash FROM images WHERE filepath = ?", (filepath_str,))
                result = cursor.fetchone()
                return result[0] if result else None
    
    def get_tags_for_image_path(self, filepath_str: str) -> list[str]:
        """获取指定图片路径的所有标签"""
        tags = []
        with self.db_lock:
            with self.conn:
                cursor = self.conn.execute('''
                    SELECT t.name
                    FROM tags t
                    JOIN image_tags it ON t.id = it.tag_id
                    JOIN images i ON it.image_id = i.id
                    WHERE i.filepath = ?
                    ORDER BY t.name
                ''', (filepath_str,))
                tags = [row[0] for row in cursor.fetchall()]
        return tags


    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
# --- End OptimizedImageDB Class ---


def _parse_input_exclude_tags(exclude_tags_str: str) -> set[str]:
    # ... (keep this function as is) ...
    if not exclude_tags_str:
        return set()
    user_tags = [t.strip() for t in exclude_tags_str.split(',') if t.strip()]
    internal_format_tags = []
    for tag in user_tags:
        internal_tag = tag.replace(' ', '_').replace('(', r'\(').replace(')', r'\)')
        internal_format_tags.append(internal_tag)
    return set([*user_tags, *internal_format_tags])


THUMBNAIL_SIZE = (128, 128)
DEFAULT_THUMBNAILS_PER_PAGE = 12
DB_SCAN_BATCH_SIZE = 100 # How many files to process before a DB bulk insert

class TaggerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Tagger and Searcher (DB Enhanced)")
        self.geometry("1100x800") # Slightly wider for DB path

        self.task_queue = queue.Queue()
        self.after(100, self.process_queue)

        self.current_interrogator: 'AbsInterrogator | None' = None

        # --- Database Initialization ---
        self.db_file_path_var = tk.StringVar(value="image_tag_cache.db") # Default DB name
        try:
            self.image_db = OptimizedImageDB(db_path=self.db_file_path_var.get())
        except Exception as e:
            messagebox.showerror("DB Error", f"Failed to initialize database: {e}\nApplication might not work correctly.")
            self.image_db = None # Indicate DB is not available
        # --- End Database Initialization ---

        self.cpu_var = tk.BooleanVar(value=False)
        # ... (other tk.Vars as before) ...
        self.rawtag_var = tk.BooleanVar(value=False)
        self.recursive_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.caption_ext_var = tk.StringVar(value=".txt")
        self.model_var = tk.StringVar()
        self.threshold_var = tk.DoubleVar(value=0.35)
        self.exclude_tags_var = tk.StringVar()
        self.gen_input_type = tk.StringVar(value="file")
        self.gen_file_path_var = tk.StringVar()
        self.gen_dir_path_var = tk.StringVar()
        
        self.search_dir_var = tk.StringVar() # Keep for "Scan to DB"
        self.search_tags_var = tk.StringVar()
        self.search_match_all_var = tk.BooleanVar(value=True) # For AND/OR search
        self.all_found_image_paths = []
        self.current_page = 0
        self.image_references = [] 
        self.search_is_active = False
        self.thumbnails_per_page_var = tk.IntVar(value=DEFAULT_THUMBNAILS_PER_PAGE)
        self.total_found_for_current_search = 0 # For DB search total

        self.current_large_preview_path: Path | None = None
        self.large_preview_photo: ImageTk.PhotoImage | None = None
        self.resize_job_id = None

        self.notebook = ttk.Notebook(self)
        self.tab_generate = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_generate, text="Generate Tags")
        self.create_tag_generation_tab() 

        self.tab_search = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_search, text="Search by Tags (DB)")
        self.create_tag_search_tab()

        self.tab_db_admin = ttk.Frame(self.notebook) # New tab for DB admin
        self.notebook.add(self.tab_db_admin, text="Database Admin")
        self.create_db_admin_tab()

        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready. Select DB and scan directories if needed.")

        if 'AbsInterrogator' not in globals() and not interrogators:
             messagebox.showwarning("Startup Warning", "Could not import 'tagger' library. Tag generation will not work.")
        if not self.image_db:
            messagebox.showerror("Startup Error", "Database could not be initialized. Search and DB Admin tabs may not function.")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)


    def on_closing(self):
        if self.image_db:
            print("Closing database connection...")
            self.image_db.close()
        self.destroy()

    def process_queue(self):
        # ... (process_queue largely same, "search_results" becomes "search_results_page") ...
        try:
            message_type, data = self.task_queue.get_nowait()
            if message_type == "status":
                self.status_var.set(data)
            elif message_type == "log_message": # Generic log for DB admin or generate
                log_text_widget = self.log_db_admin_text if self.notebook.select().endswith("db_admin") else self.log_generate_text
                log_text_widget.config(state=tk.NORMAL)
                log_text_widget.insert(tk.END, data + "\n")
                log_text_widget.see(tk.END)
                log_text_widget.config(state=tk.DISABLED)
            elif message_type == "error":
                messagebox.showerror("Error", data)
            elif message_type == "search_results_page": 
                new_paths = data
                is_first_batch = not self.all_found_image_paths
                self.all_found_image_paths.extend(new_paths)
                if is_first_batch and self.all_found_image_paths:
                    self.current_page = 1
                    self.show_page(self.current_page) 
                elif not self.all_found_image_paths and not self.search_is_active: # No results at all
                    self.clear_large_preview("No images found matching your criteria.")
                    for widget in self.scrollable_frame.winfo_children(): widget.destroy()
                    ttk.Label(self.scrollable_frame, text="No images found.").pack(padx=10, pady=10)

                self.update_pagination_controls()

            elif message_type == "search_complete": 
                self.total_found_for_current_search = data # This is the total count from DB
                self.search_is_active = False
                if self.total_found_for_current_search == 0 and not self.all_found_image_paths:
                    for widget in self.scrollable_frame.winfo_children(): widget.destroy()
                    self.image_references = []
                    ttk.Label(self.scrollable_frame, text="No images found matching your criteria.").pack(padx=10, pady=10)
                    self.clear_large_preview("No images found.")
                self.update_pagination_controls() 
            elif message_type == "enable_generate_button":
                self.generate_button.config(state=tk.NORMAL if interrogators and 'AbsInterrogator' in globals() else tk.DISABLED)
            elif message_type == "enable_search_button":
                self.search_button.config(state=tk.NORMAL)
            elif message_type == "enable_db_scan_button":
                self.scan_dir_to_db_button.config(state=tk.NORMAL)

        except queue.Empty:
            pass
        self.after(100, self.process_queue)


    def create_tag_generation_tab(self):
        # ... (This tab remains largely the same) ...
        frame = ttk.Frame(self.tab_generate, padding="10")
        frame.pack(expand=True, fill="both")

        input_frame = ttk.LabelFrame(frame, text="Input Source")
        input_frame.pack(fill=tk.X, pady=5)

        self.file_button = ttk.Button(input_frame, text="Select Image File", command=self.select_image_file)
        self.file_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.gen_file_label = ttk.Label(input_frame, textvariable=self.gen_file_path_var, width=50, wraplength=300)
        self.gen_file_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.dir_button = ttk.Button(input_frame, text="Select Image Directory", command=self.select_image_dir_gen)
        self.dir_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        options_frame = ttk.LabelFrame(frame, text="Options")
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(options_frame, text="Model:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        model_list = list(interrogators.keys()) if interrogators else ["(No models found)"]
        default_model = model_list[0] if model_list and model_list[0] != "(No models found)" else ""
        self.model_var.set(default_model) 
        self.model_menu = ttk.Combobox(options_frame, textvariable=self.model_var, values=model_list, state="readonly" if default_model else "disabled")
        self.model_menu.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        self.model_menu.bind("<<ComboboxSelected>>", self.load_interrogator_model)

        ttk.Label(options_frame, text="Threshold:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.threshold_entry = ttk.Entry(options_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(options_frame, text="Exclude Tags (comma-sep):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.exclude_tags_entry = ttk.Entry(options_frame, textvariable=self.exclude_tags_var)
        self.exclude_tags_entry.grid(row=2, column=1, columnspan=3, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(options_frame, text="Caption File Ext:").grid(row=0, column=2, padx=(10,5), pady=2, sticky=tk.W)
        self.caption_ext_entry = ttk.Entry(options_frame, textvariable=self.caption_ext_var, width=10)
        self.caption_ext_entry.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)

        checkbox_frame = ttk.Frame(options_frame)
        checkbox_frame.grid(row=3, column=0, columnspan=4, pady=5, sticky=tk.W)

        ttk.Checkbutton(checkbox_frame, text="CPU Only", variable=self.cpu_var, command=self.update_interrogator_cpu).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checkbox_frame, text="Raw Tags (no escape/space)", variable=self.rawtag_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checkbox_frame, text="Recursive (for Dir)", variable=self.recursive_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checkbox_frame, text="Overwrite .txt (for Dir)", variable=self.overwrite_var).pack(side=tk.LEFT, padx=5)

        options_frame.columnconfigure(1, weight=1)

        self.generate_button = ttk.Button(frame, text="Generate Tags", command=self.start_generate_tags, 
                                          state="disabled") 
        self.generate_button.pack(pady=10)

        if default_model: 
            self.load_interrogator_model() 

        log_frame = ttk.LabelFrame(frame, text="Log (Tag Generation)")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_generate_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_generate_text.yview)
        self.log_generate_text.config(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_generate_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


    def create_db_admin_tab(self):
        frame = ttk.Frame(self.tab_db_admin, padding="10")
        frame.pack(expand=True, fill="both")

        # DB File Path
        db_path_frame = ttk.LabelFrame(frame, text="Database File")
        db_path_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(db_path_frame, text="Path:").pack(side=tk.LEFT, padx=5, pady=5)
        db_path_entry = ttk.Entry(db_path_frame, textvariable=self.db_file_path_var, width=60)
        db_path_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        ttk.Button(db_path_frame, text="Browse/Set", command=self.select_db_file).pack(side=tk.LEFT, padx=5, pady=5)

        # Scan Directory to DB
        scan_frame = ttk.LabelFrame(frame, text="Scan Directory to Database")
        scan_frame.pack(fill=tk.X, pady=10)

        self.db_scan_dir_button = ttk.Button(scan_frame, text="Select Directory to Scan", command=self.select_scan_dir_for_db)
        self.db_scan_dir_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Using self.search_dir_var for the directory to scan into DB
        self.db_scan_dir_label = ttk.Label(scan_frame, textvariable=self.search_dir_var, width=50, wraplength=350)
        self.db_scan_dir_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.scan_dir_to_db_button = ttk.Button(scan_frame, text="Start Scan & Add to DB", command=self.start_scan_directory_to_db, state=tk.DISABLED)
        self.scan_dir_to_db_button.pack(pady=10, padx=5)
        if self.search_dir_var.get() and self.image_db: # Enable if dir is pre-set and DB is up
            self.scan_dir_to_db_button.config(state=tk.NORMAL)


        # Log for DB Admin
        log_frame = ttk.LabelFrame(frame, text="Log (Database Operations)")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_db_admin_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_db_admin_text.yview)
        self.log_db_admin_text.config(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_db_admin_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def select_db_file(self):
        filepath = filedialog.asksaveasfilename(
            title="Select or Create Database File",
            defaultextension=".db",
            filetypes=(("SQLite Database", "*.db"), ("All files", "*.*"))
        )
        if filepath:
            self.db_file_path_var.set(filepath)
            try:
                if self.image_db:
                    self.image_db.close()
                self.image_db = OptimizedImageDB(db_path=filepath)
                self.task_queue.put(("status", f"Database set to: {filepath}"))
                self.task_queue.put(("log_message", f"Database initialized at: {filepath}"))
                if self.search_dir_var.get(): # Re-enable scan button if dir is already selected
                     self.scan_dir_to_db_button.config(state=tk.NORMAL)
            except Exception as e:
                self.image_db = None
                messagebox.showerror("DB Error", f"Failed to set new database: {e}")
                self.task_queue.put(("status", "Error setting database."))
                self.scan_dir_to_db_button.config(state=tk.DISABLED)


    def select_scan_dir_for_db(self):
        dirpath = filedialog.askdirectory(title="Select Directory to Scan into Database")
        if dirpath:
            self.search_dir_var.set(dirpath) # Re-use this var for the scan target
            self.task_queue.put(("log_message", f"Directory to scan: {dirpath}"))
            if self.image_db : # Only enable if DB is also working
                self.scan_dir_to_db_button.config(state=tk.NORMAL)
            else:
                self.scan_dir_to_db_button.config(state=tk.DISABLED)
                self.task_queue.put(("log_message", "Database not initialized. Select DB file first."))


    def start_scan_directory_to_db(self):
        if not self.image_db:
            messagebox.showerror("DB Error", "Database is not initialized. Cannot scan.")
            return
        
        scan_dir = self.search_dir_var.get()
        if not scan_dir:
            messagebox.showerror("Input Error", "Please select a directory to scan.")
            return

        self.scan_dir_to_db_button.config(state=tk.DISABLED)
        self.task_queue.put(("status", f"Scanning directory {Path(scan_dir).name} to DB..."))
        self.log_db_admin_text.config(state=tk.NORMAL); self.log_db_admin_text.delete('1.0', tk.END); self.log_db_admin_text.config(state=tk.DISABLED)


        thread = threading.Thread(target=self._scan_dir_to_db_thread, args=(scan_dir,))
        thread.daemon = True
        thread.start()

    def _scan_dir_to_db_thread(self, dir_to_scan_str: str):
        # This is a conceptual placeholder for the actual scanning and DB insertion logic
        # It needs to:
        # 1. Walk the directory_path.
        # 2. For each image, calculate hash.
        # 3. Read associated tag file (e.g., .txt based on caption_ext_var).
        # 4. Prepare data for OptimizedImageDB.bulk_insert.
        # 5. Call bulk_insert in batches.
        self.task_queue.put(("log_message", f"Starting DB scan for: {dir_to_scan_str}"))
        root_path = Path(dir_to_scan_str)
        images_data_batch = []
        processed_count = 0
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp'] # Common image extensions
        caption_ext = self.caption_ext_var.get()
        if not caption_ext.startswith('.'): caption_ext = '.' + caption_ext

        try:
            for item in root_path.rglob("*"): # Recursive glob
                if item.is_file() and item.suffix.lower() in image_extensions:
                    image_path = item
                    tag_file_path = image_path.with_suffix(caption_ext)
                    tags = []

                    if tag_file_path.exists():
                        try:
                            with open(tag_file_path, 'r', encoding='utf-8') as tf:
                                content = tf.read()
                                tags = [t.strip() for t in content.split(',') if t.strip()]
                        except Exception as e:
                            self.task_queue.put(("log_message", f"Warn: Could not read/parse tag file {tag_file_path.name}: {e}"))
                    
                    file_hash = self.image_db.calculate_file_hash(str(image_path))
                    if not file_hash:
                        self.task_queue.put(("log_message", f"Warn: Could not hash {image_path.name}, skipping."))
                        continue
                    
                    file_size = image_path.stat().st_size
                    
                    images_data_batch.append({
                        'image': (image_path.name, str(image_path), file_size, file_hash),
                        'tags': tags
                    })
                    processed_count += 1

                    if len(images_data_batch) >= DB_SCAN_BATCH_SIZE:
                        self.image_db.bulk_insert(images_data_batch)
                        self.task_queue.put(("log_message", f"Inserted batch of {len(images_data_batch)} items. Total processed: {processed_count}"))
                        images_data_batch.clear()
            
            if images_data_batch: # Insert any remaining items
                self.image_db.bulk_insert(images_data_batch)
                self.task_queue.put(("log_message", f"Inserted final batch of {len(images_data_batch)} items. Total processed: {processed_count}"))

            self.task_queue.put(("log_message", f"DB Scan complete for {dir_to_scan_str}. Processed {processed_count} images."))
            self.task_queue.put(("status", "DB Scan complete."))
        except Exception as e:
            self.task_queue.put(("log_message", f"ERROR during DB scan: {e}"))
            self.task_queue.put(("status", "Error during DB scan."))
            import traceback
            self.task_queue.put(("log_message", f"TRACEBACK: {traceback.format_exc()}"))
        finally:
            self.task_queue.put(("enable_db_scan_button", None))


    # --- Methods for Tag Generation Tab (select_image_file, etc.) ---
    # These remain largely the same
    def select_image_file(self):
        # ... (same as before)
        filepath = filedialog.askopenfilename(title="Select Image File", filetypes=(("Image files", "*.png *.jpg .jpeg .webp"), ("All files", "*.*")))
        if filepath:
            self.gen_input_type.set("file")
            self.gen_file_path_var.set(filepath)
            self.gen_dir_path_var.set("")
            self.task_queue.put(("log_message", f"Selected image for tag gen: {Path(filepath).name}"))

    def select_image_dir_gen(self):
        # ... (same as before)
        dirpath = filedialog.askdirectory(title="Select Image Directory for Tag Gen")
        if dirpath:
            self.gen_input_type.set("dir")
            self.gen_dir_path_var.set(dirpath)
            self.gen_file_path_var.set("")
            self.task_queue.put(("log_message", f"Selected directory for tag gen: {dirpath}"))


    def load_interrogator_model(self, event=None):
        # ... (same as before, but use log_message)
        model_name = self.model_var.get()
        if model_name and model_name != "(No models found)" and model_name in interrogators:
            self.current_interrogator = interrogators[model_name]
            self.task_queue.put(("log_message", f"Loaded model: {model_name}"))
            self.update_interrogator_cpu()
            self.generate_button.config(state="normal" if 'AbsInterrogator' in globals() else "disabled")
        elif model_name == "(No models found)":
            self.task_queue.put(("log_message", "No models available to load."))
            self.generate_button.config(state="disabled")
        else:
            self.current_interrogator = None
            self.task_queue.put(("error", f"Model '{model_name}' not found!"))
            self.generate_button.config(state="disabled")


    def update_interrogator_cpu(self):
        # ... (same as before, but use log_message)
        if self.current_interrogator and hasattr(self.current_interrogator, 'use_cpu'): 
            if self.cpu_var.get():
                try:
                    self.current_interrogator.use_cpu()
                    self.task_queue.put(("log_message", f"Using CPU for model {self.model_var.get()}."))
                except Exception as e:
                    self.task_queue.put(("error", f"Error setting model to CPU: {e}"))
            else:
                if hasattr(self.current_interrogator, 'use_gpu'):
                    try:
                        self.current_interrogator.use_gpu()
                        self.task_queue.put(("log_message", f"Attempting to use GPU for model {self.model_var.get()} (if available)."))
                    except Exception as e:
                         self.task_queue.put(("log_message", f"Note: Could not explicitly switch to GPU for {self.model_var.get()}: {e} (May use default)."))
                else: 
                    self.task_queue.put(("log_message", f"GPU usage (if available by default) for model {self.model_var.get()}."))

    def start_generate_tags(self):
        # ... (same as before, but use log_message)
        if 'AbsInterrogator' not in globals():
            messagebox.showerror("Error", "AbsInterrogator class not loaded. Cannot process tags.")
            return
        if not self.current_interrogator:
            messagebox.showerror("Error", "No model selected or model failed to load.")
            return

        input_type = self.gen_input_type.get()
        file_path_str = self.gen_file_path_var.get()
        dir_path_str = self.gen_dir_path_var.get()

        if input_type == "file" and not file_path_str:
            messagebox.showerror("Error", "Please select an image file.")
            return
        if input_type == "dir" and not dir_path_str:
            messagebox.showerror("Error", "Please select a directory.")
            return

        self.generate_button.config(state=tk.DISABLED)
        self.task_queue.put(("status", "Generating tags..."))
        self.log_generate_text.config(state=tk.NORMAL); self.log_generate_text.delete('1.0', tk.END); self.log_generate_text.config(state=tk.DISABLED)

        threshold = self.threshold_var.get()
        use_postprocessing_rules = not self.rawtag_var.get()
        exclude_tags_set = _parse_input_exclude_tags(self.exclude_tags_var.get())
        caption_ext = self.caption_ext_var.get()
        if not caption_ext.startswith('.'): caption_ext = '.' + caption_ext

        thread = threading.Thread(target=self._generate_tags_thread, args=(
            input_type, file_path_str, dir_path_str,
            threshold, use_postprocessing_rules, exclude_tags_set,
            self.recursive_var.get(), self.overwrite_var.get(), caption_ext
        ))
        thread.daemon = True
        thread.start()


    def _generate_tags_thread(self, input_type, file_path_str, dir_path_str,
                            threshold_val, use_postprocessing, exclude_tags_set,
                            recursive_flag, overwrite_flag, caption_ext_val):
        # ... (same as before, use log_message, and optionally add generated tags to DB)
        if 'AbsInterrogator' not in globals():
            self.task_queue.put(("error", "Tag generation thread: AbsInterrogator not available."))
            self.task_queue.put(("status", "Error: Tagger library component missing."))
            self.task_queue.put(("enable_generate_button", None)) 
            return

        generated_data_for_db = []

        try:
            def _explore_image_files_recursive(folder_path: Path, is_recursive: bool) -> Generator[Path, None, None]:
                for item in folder_path.iterdir():
                    if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                        yield item
                    elif is_recursive and item.is_dir():
                        yield from _explore_image_files_recursive(item, is_recursive)

            if input_type == "file":
                image_path = Path(file_path_str)
                self.task_queue.put(("log_message", f"Processing for tags: {image_path}"))
                im = Image.open(image_path).convert("RGB")
                raw_model_output = self.current_interrogator.interrogate(im)
                tags_from_model = {}
                if isinstance(raw_model_output, tuple):
                    if len(raw_model_output) == 3 and isinstance(raw_model_output[2], dict): 
                        tags_from_model = raw_model_output[2]
                    elif len(raw_model_output) == 2 and isinstance(raw_model_output[1], dict): 
                        tags_from_model = raw_model_output[1]
                elif isinstance(raw_model_output, dict): 
                    tags_from_model = raw_model_output
                
                if not tags_from_model:
                    self.task_queue.put(("error", f"Unexpected or empty model output for {image_path.name}. Got: {type(raw_model_output)}"))

                processed_tags_dict = AbsInterrogator.postprocess_tags(
                    tags_from_model, threshold=threshold_val, escape_tag=use_postprocessing, 
                    replace_underscore=use_postprocessing, exclude_tags=exclude_tags_set
                )
                tags_list = list(processed_tags_dict.keys())
                tags_str = ', '.join(tags_list)
                self.task_queue.put(("log_message", f"\n--- Tags for {image_path.name} ---\n{tags_str}\n--- End of Tags ---"))
                
                # Prepare for DB
                if self.image_db:
                    file_hash = self.image_db.calculate_file_hash(str(image_path))
                    if file_hash:
                        generated_data_for_db.append({
                            'image': (image_path.name, str(image_path), image_path.stat().st_size, file_hash),
                            'tags': tags_list
                        })


            elif input_type == "dir":
                root_path = Path(dir_path_str)
                for image_path in _explore_image_files_recursive(root_path, recursive_flag):
                    caption_path = image_path.parent / f'{image_path.stem}{caption_ext_val}'
                    if caption_path.is_file() and not overwrite_flag:
                        self.task_queue.put(("log_message", f"Skip (exists): {image_path.name}"))
                        continue
                    self.task_queue.put(("log_message", f"Processing for tags: {image_path.name}"))
                    try:
                        im = Image.open(image_path).convert("RGB")
                        raw_model_output = self.current_interrogator.interrogate(im)
                        tags_from_model = {}
                        if isinstance(raw_model_output, tuple):
                            if len(raw_model_output) == 3 and isinstance(raw_model_output[2], dict):
                                tags_from_model = raw_model_output[2]
                            elif len(raw_model_output) == 2 and isinstance(raw_model_output[1], dict):
                                tags_from_model = raw_model_output[1]
                        elif isinstance(raw_model_output, dict):
                            tags_from_model = raw_model_output

                        if not tags_from_model:
                            self.task_queue.put(("log_message", f"WARN: Unexpected or empty model output for {image_path.name}, skipping tag saving for this file."))
                            continue

                        processed_tags_dict = AbsInterrogator.postprocess_tags(
                            tags_from_model, threshold=threshold_val, escape_tag=use_postprocessing,
                            replace_underscore=use_postprocessing, exclude_tags=exclude_tags_set
                        )
                        tags_list = list(processed_tags_dict.keys())
                        tags_str = ', '.join(tags_list)
                        with open(caption_path, 'w', encoding='utf-8') as fp: fp.write(tags_str)
                        self.task_queue.put(("log_message", f"Saved tags to: {caption_path.name}"))

                        # Prepare for DB
                        if self.image_db:
                            file_hash = self.image_db.calculate_file_hash(str(image_path))
                            if file_hash:
                                generated_data_for_db.append({
                                    'image': (image_path.name, str(image_path), image_path.stat().st_size, file_hash),
                                    'tags': tags_list
                                })
                                if len(generated_data_for_db) >= DB_SCAN_BATCH_SIZE:
                                    self.image_db.bulk_insert(generated_data_for_db)
                                    self.task_queue.put(("log_message", f"Added {len(generated_data_for_db)} generated tags to DB."))
                                    generated_data_for_db.clear()

                    except Exception as e_inner:
                        self.task_queue.put(("log_message", f"ERROR processing {image_path.name}: {e_inner}"))
            
            if self.image_db and generated_data_for_db: # Add remaining to DB
                self.image_db.bulk_insert(generated_data_for_db)
                self.task_queue.put(("log_message", f"Added final batch of {len(generated_data_for_db)} generated tags to DB."))


            self.task_queue.put(("status", "Tag generation complete."))
        except Exception as e_outer:
            self.task_queue.put(("status", "Error during tag generation."))
            self.task_queue.put(("error", f"An error occurred in generation thread: {e_outer}"))
            import traceback
            self.task_queue.put(("log_message", f"TRACEBACK: {traceback.format_exc()}"))
        finally:
            self.task_queue.put(("enable_generate_button", None))


    def create_tag_search_tab(self):
        frame = ttk.Frame(self.tab_search, padding="10")
        frame.pack(expand=True, fill="both")

        # --- Top Controls ---
        top_controls_frame = ttk.Frame(frame)
        top_controls_frame.pack(fill=tk.X, pady=5, side=tk.TOP) # Explicitly pack to TOP

        search_bar_frame = ttk.Frame(top_controls_frame)
        search_bar_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Label(search_bar_frame, text="Search Tags (comma-sep):").pack(side=tk.LEFT, padx=5)
        self.search_tags_entry = ttk.Entry(search_bar_frame, textvariable=self.search_tags_var, width=50)
        self.search_tags_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.search_match_all_checkbox = ttk.Checkbutton(search_bar_frame, text="Match All (AND)", variable=self.search_match_all_var)
        self.search_match_all_checkbox.pack(side=tk.LEFT, padx=5)

        self.search_button = ttk.Button(search_bar_frame, text="Search DB", command=self.start_search_tags)
        self.search_button.pack(side=tk.LEFT, padx=5)

        # --- Bottom Controls (Packed BEFORE PanedWindow in vertical stacking) ---
        bottom_controls_frame = ttk.Frame(frame)
        bottom_controls_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM) # Explicitly pack to BOTTOM

        pagination_frame = ttk.Frame(bottom_controls_frame)
        pagination_frame.pack(side=tk.LEFT, fill=tk.X, expand=True) # Allow pagination to expand
        self.prev_page_button = ttk.Button(pagination_frame, text="<< Previous", command=self.prev_page, state=tk.DISABLED)
        self.prev_page_button.pack(side=tk.LEFT, padx=5)
        self.page_label_var = tk.StringVar(value="Page 0/0")
        ttk.Label(pagination_frame, textvariable=self.page_label_var).pack(side=tk.LEFT, padx=5)
        self.next_page_button = ttk.Button(pagination_frame, text="Next >>", command=self.next_page, state=tk.DISABLED)
        self.next_page_button.pack(side=tk.LEFT, padx=5)

        settings_frame = ttk.Frame(bottom_controls_frame)
        settings_frame.pack(side=tk.RIGHT, padx=5) # Keep settings to the right of pagination
        ttk.Label(settings_frame, text="Thumbs/page:").pack(side=tk.LEFT)
        self.thumbs_per_page_spinbox = ttk.Spinbox(
            settings_frame,
            from_=4, to=100, 
            textvariable=self.thumbnails_per_page_var,
            width=5,
            command=self.on_thumbnails_per_page_changed
        )
        self.thumbs_per_page_spinbox.bind("<Return>", lambda e: self.on_thumbnails_per_page_changed())
        self.thumbs_per_page_spinbox.pack(side=tk.LEFT, padx=5)
        
        # --- Main Results Area with PanedWindow (Packed AFTER bottom controls to fill remaining space) ---
        results_outer_frame = ttk.LabelFrame(frame, text="Search Results from Database")
        # This will now fill the space between top_controls_frame and bottom_controls_frame
        results_outer_frame.pack(fill=tk.BOTH, expand=True, pady=5, side=tk.TOP) 

        self.paned_window = ttk.PanedWindow(results_outer_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Left Pane for Thumbnails
        self.left_pane_frame = ttk.Frame(self.paned_window)
        self.canvas = tk.Canvas(self.left_pane_frame)
        self.scrollbar = ttk.Scrollbar(self.left_pane_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        # Mousewheel bindings should ideally be on the canvas or specific scrollable widgets,
        # not bind_all, to avoid conflicts if other scrollable areas exist.
        # For now, assuming this is the primary scroll area in this tab.
        self.canvas.bind("<MouseWheel>", self._on_mousewheel) # For Windows/macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel) # For Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel) # For Linux scroll down

        self.paned_window.add(self.left_pane_frame, weight=1) 

        # Right Pane for Large Preview
        self.right_pane_frame = ttk.Frame(self.paned_window)
        self.large_preview_label = ttk.Label(self.right_pane_frame, text="Click a thumbnail to preview", anchor=tk.CENTER, justify=tk.CENTER)
        self.large_preview_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.paned_window.add(self.right_pane_frame, weight=3) 
        self.right_pane_frame.bind("<Configure>", self.on_right_pane_configure)

    def on_thumbnails_per_page_changed(self):
        # ... (same as before) ...
        try:
            new_val = self.thumbnails_per_page_var.get()
        except tk.TclError:
            self.task_queue.put(("error", "Invalid value for thumbnails per page."))
            self.thumbnails_per_page_var.set(DEFAULT_THUMBNAILS_PER_PAGE)
            return

        self.task_queue.put(("status", f"Thumbnails per page set to {new_val}."))
        if self.all_found_image_paths: # If results are already displayed, re-trigger search or re-paginate
            # For simplicity, let's re-trigger the search if one was active
            # Or, if not active, just re-display current page.
            # A full re-search might be too much. Let's just re-paginate current `all_found_image_paths`.
            # However, since DB search is page-based, we *should* re-search.
            if self.search_tags_var.get().strip(): # If there was a previous search query
                self.start_search_tags() # Re-issue search with new page size
            else: # Just update view if no active query but items somehow loaded
                 self.current_page = 1 
                 self.show_page(self.current_page)
                 self.update_pagination_controls()


    def _on_mousewheel(self, event):
        # ... (same as before) ...
        if event.num == 4 or event.delta > 0: 
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0: 
            self.canvas.yview_scroll(1, "units")

    # select_search_dir is now select_scan_dir_for_db, no directory needed for search itself

    def start_search_tags(self):
        if not self.image_db:
            messagebox.showerror("DB Error", "Database is not initialized. Cannot search.")
            return

        search_query_str = self.search_tags_var.get()
        if not search_query_str.strip():
            messagebox.showinfo("Search", "Please enter tags to search for.")
            # Clear previous results if any
            self.all_found_image_paths = []
            self.current_page = 0
            self.total_found_for_current_search = 0
            for widget in self.scrollable_frame.winfo_children(): widget.destroy()
            self.image_references = []
            self.clear_large_preview("Enter search tags.")
            self.update_pagination_controls()
            return
        
        self.search_button.config(state=tk.DISABLED)
        self.task_queue.put(("status", "Searching database..."))

        self.all_found_image_paths = [] # Clear previous results
        self.current_page = 0
        self.total_found_for_current_search = 0 # Reset total for new search
        self.search_is_active = True
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.image_references = []
        self.clear_large_preview("Searching DB...")
        self.update_pagination_controls() # Update to "Searching..." state

        thread = threading.Thread(target=self._search_tags_thread, args=(search_query_str,))
        thread.daemon = True
        thread.start()

    def _search_tags_thread(self, search_query_str: str):
        current_db_page = 1
        initial_total_count_set = False
        
        # Parse search_query_str into a list of tags
        query_tags_list = [tag.strip().lower() for tag in search_query_str.split(',') if tag.strip()]

        if not query_tags_list:
            self.task_queue.put(("search_results_page", []))
            self.task_queue.put(("search_complete", 0))
            self.task_queue.put(("status", "Search complete. Found 0 images for empty query."))
            self.task_queue.put(("enable_search_button", None))
            return

        try:
            while self.search_is_active: # Loop controlled by GUI (e.g. new search starts) or no more results
                page_results, total_items_from_db = self.image_db.search(
                    query_tags=query_tags_list,
                    page=current_db_page,
                    page_size=self.thumbnails_per_page_var.get(),
                    match_all=self.search_match_all_var.get()
                )

                if not initial_total_count_set:
                    # Send total count once with the first batch of results or if no results
                    self.task_queue.put(("search_complete", total_items_from_db))
                    initial_total_count_set = True # total_found_for_current_search will be set in process_queue

                if page_results:
                    self.task_queue.put(("search_results_page", page_results))
                    current_db_page += 1
                else: # No more results for this query from the DB
                    # If total_items_from_db was 0 initially, search_complete(0) was already sent.
                    # If there were results, search_complete(total) was sent.
                    # Now, just break the loop.
                    break
            
            # Final status update (search_complete message already handles total count)
            # Total count comes from the search_complete message data
            # self.task_queue.put(("status", f"DB Search finished.")) # Status set by search_complete

        except Exception as e_outer:
            self.task_queue.put(("status", "Error during DB search."))
            self.task_queue.put(("error", f"DB Search error: {e_outer}"))
            import traceback
            self.task_queue.put(("log_message", f"TRACEBACK (DB Search Thread): {traceback.format_exc()}"))
            if not initial_total_count_set: # If error happened before first result
                self.task_queue.put(("search_complete", 0))
        finally:
            self.search_is_active = False # Ensure flag is reset
            self.task_queue.put(("enable_search_button", None))


    def clear_large_preview(self, message="Click a thumbnail to preview"):
        # ... (same as before) ...
        self.large_preview_label.config(image='', text=message)
        self.large_preview_photo = None
        self.current_large_preview_path = None

    def display_large_preview(self, image_path: Path):
        # ... (same as before, ensure log_message for errors) ...
        if not image_path or not image_path.is_file():
            self.clear_large_preview("Image not found.")
            return

        self.current_large_preview_path = image_path 

        try:
            img = Image.open(image_path)
            self.right_pane_frame.update_idletasks() 
            preview_max_width = self.right_pane_frame.winfo_width() - 10
            preview_max_height = self.right_pane_frame.winfo_height() - 10

            if preview_max_width <= 1 or preview_max_height <= 1:
                preview_max_width, preview_max_height = 400, 400 

            img_copy = img.copy() 
            img_copy.thumbnail((preview_max_width, preview_max_height), RESAMPLING_METHOD)
            
            photo = ImageTk.PhotoImage(img_copy)
            self.large_preview_photo = photo 
            self.large_preview_label.config(image=self.large_preview_photo, text="")
            self.large_preview_label.image = self.large_preview_photo

        except Exception as e:
            error_message = f"Error previewing:\n{image_path.name}\n{e}"
            self.clear_large_preview(error_message)
            print(f"Error displaying large preview for {image_path}: {e}")
            self.task_queue.put(("log_message", f"Error previewing {image_path.name}: {e}"))


    def on_right_pane_configure(self, event):
        # ... (same as before) ...
        if self.resize_job_id:
            self.after_cancel(self.resize_job_id)
        if self.current_large_preview_path and \
           self.right_pane_frame.winfo_width() > 10 and \
           self.right_pane_frame.winfo_height() > 10:
            self.resize_job_id = self.after(250, self._do_resize_large_preview)

    def _do_resize_large_preview(self):
        # ... (same as before) ...
        self.resize_job_id = None
        if self.current_large_preview_path: 
            self.display_large_preview(self.current_large_preview_path)

    def show_page(self, page_num):
        # ... (same as before, ensure log_message for errors) ...
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.image_references = [] 
        self.current_page = page_num
        
        thumbs_per_page = self.thumbnails_per_page_var.get()
        start_index = (self.current_page - 1) * thumbs_per_page
        end_index = start_index + thumbs_per_page
        page_image_paths = self.all_found_image_paths[start_index:end_index]

        cols = 4 
        thumb_container_width = THUMBNAIL_SIZE[0] + 10 

        for i, img_path in enumerate(page_image_paths): # img_path is already Path object
            item_frame = ttk.Frame(self.scrollable_frame)
            row, col = i // cols, i % cols
            item_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            try:
                img = Image.open(img_path)
                img.thumbnail(THUMBNAIL_SIZE, RESAMPLING_METHOD)
                photo = ImageTk.PhotoImage(img)
                self.image_references.append(photo) 

                thumb_label = ttk.Label(item_frame, image=photo, relief="solid", borderwidth=1)
                thumb_label.image = photo 
                thumb_label.pack(pady=(0,2))
                thumb_label.bind("<Button-1>", lambda e, p=img_path: self.display_large_preview(p))
                thumb_label.bind("<Double-1>", lambda e, p=img_path: self.open_image_viewer(p))

                filename_label = ttk.Label(item_frame, text=img_path.name, anchor=tk.CENTER, wraplength=thumb_container_width - 10)
                filename_label.pack(fill=tk.X)

            except Exception as e_inner:
                err_text = f"Error:\n{img_path.name[:20]}...\n(Corrupt?)"
                err_label = ttk.Label(item_frame, text=err_text, relief="solid", borderwidth=1, anchor=tk.CENTER, justify=tk.CENTER, width=int(THUMBNAIL_SIZE[0]/7), height=int(THUMBNAIL_SIZE[1]/15) )
                err_label.pack(pady=(0,2), fill=tk.BOTH, expand=True)
                print(f"Error loading thumbnail for {img_path}: {e_inner}")
                self.task_queue.put(("log_message", f"Error loading thumbnail for {img_path}: {e_inner}"))

        for c_idx in range(cols): 
            self.scrollable_frame.columnconfigure(c_idx, weight=1, minsize=thumb_container_width)
        
        self.update_pagination_controls()
        self.canvas.yview_moveto(0) 


    def update_pagination_controls(self):
        thumbs_per_page = self.thumbnails_per_page_var.get()
        if thumbs_per_page <= 0: thumbs_per_page = DEFAULT_THUMBNAILS_PER_PAGE # Safety

        # Use the total count received from the database search
        current_total_items = self.total_found_for_current_search
        current_total_pages = (current_total_items + thumbs_per_page - 1) // thumbs_per_page
        if current_total_pages == 0 and current_total_items > 0 : # Should be 1 if items exist
            current_total_pages = 1
        
        if current_total_items == 0 and not self.search_is_active: # No items and search not running
             self.page_label_var.set("Page 0/0")
        elif self.search_is_active and current_total_items == 0 and not self.all_found_image_paths: # Searching, no known total yet, no results yet
             self.page_label_var.set("Page -/- (Searching...)")
        else:
            self.page_label_var.set(f"Page {self.current_page}/{current_total_pages}")

        self.prev_page_button.config(state=tk.NORMAL if self.current_page > 1 else tk.DISABLED)
        # Enable Next if current page < total pages OR if search is active (more pages might be streamed from thread, though DB search gives total upfront)
        # With DB giving total count upfront, `search_is_active` for enabling Next is less critical.
        self.next_page_button.config(state=tk.NORMAL if self.current_page < current_total_pages else tk.DISABLED)


    def prev_page(self):
        # ... (same as before) ...
        if self.current_page > 1:
            self.show_page(self.current_page - 1)

    def next_page(self):
        # ... (same as before, relies on update_pagination_controls for correct state) ...
        thumbs_per_page = self.thumbnails_per_page_var.get()
        # Total pages calculated using self.total_found_for_current_search
        current_max_pages = (self.total_found_for_current_search + thumbs_per_page - 1) // thumbs_per_page
        if self.current_page < current_max_pages:
            self.show_page(self.current_page + 1)


    def open_image_viewer(self, image_path: Path):
        # ... (same as before) ...
        self.task_queue.put(("status", f"Opening {image_path.name}..."))
        try:
            image_path_str = str(image_path.resolve()) 
            if sys.platform == "win32":
                os.startfile(image_path_str)
            elif sys.platform == "darwin": 
                subprocess.run(["open", image_path_str], check=True)
            else: 
                subprocess.run(["xdg-open", image_path_str], check=True)
        except FileNotFoundError:
             messagebox.showerror("Error", f"Could not open image: '{image_path_str}' not found.")
             self.task_queue.put(("status", f"Failed to open {image_path.name}, file not found."))
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image '{image_path_str}'.\n{e}")
            self.task_queue.put(("status", f"Failed to open {image_path.name}."))


if __name__ == '__main__':
    app = TaggerApp()
    app.mainloop()
