import pandas as pd
from ydata_profiling import ProfileReport
import ttkbootstrap as ttk
import customtkinter as ctk
from tkinter import messagebox, filedialog, ttk as tk_tt
import os, webbrowser
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # Added Regressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR # Added SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge # Added LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import joblib
import final_pipeline as fcp
from scipy.stats import gmean
import numpy as np # Import numpy for isnumeric check

# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")  # "light", "dark", "system"
ctk.set_default_color_theme("blue")  # Other available themes: "green", "dark-blue"

class CSVProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoML Pipeline - ModelXpert") # More general title
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Using ttkbootstrap theme
        self.style = ttk.Style()
        self.style.theme_use("cosmo")

        # Data storage
        self.df_original = None
        self.df_processed = None
        self.global_model_results = None # For classification results
        self.global_regression_results = None # For regression results
        self.task_type = ctk.StringVar(value="Classification") # To store user choice: Classification or Regression
        self.label_var = ctk.StringVar() # Changed to ctk.StringVar for consistency
        self.feature_vars = [] # List to store feature selection variables

        # Create pages
        self.create_pages()
        # Create navigation bar
        self.create_nav_bar()

        # Show Upload Page first
        self.show_page(self.page_upload)

    def create_nav_bar(self):
        """Top navigation bar for easy page switching."""
        self.nav_bar = ctk.CTkFrame(self.root, height=50)
        self.nav_bar.pack(side="top", anchor="center", pady=(10, 0))

        btn_style = {"width": 180,
                     "height": 35,
                     "corner_radius": 6,
                     "fg_color": "#009990",
                     "text_color": "#001A6E",
                     "hover_color": "#E1FFBB",
                     "font": ("Helvetica", 14, "bold")
                     }

        ctk.CTkButton(
            self.nav_bar, text="Upload CSV", command=lambda: self.show_page(self.page_upload), **btn_style
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="View Data", command=self.safe_show_page_data_view, **btn_style # Added safety check
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="Select Columns", command=self.safe_show_page_column_select, **btn_style # Added safety check
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="Results", command=self.safe_show_page_results, **btn_style # Added safety check
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="Create Data Viz", command=self.generate_ydata_report, **btn_style # Renamed slightly
        ).pack(side="left", padx=5)

    def create_pages(self):
        """Initialize all pages."""
        self.page_upload = ctk.CTkFrame(self.root)
        self.page_data_view = ctk.CTkFrame(self.root)
        self.page_column_select = ctk.CTkFrame(self.root)
        self.page_results = ctk.CTkFrame(self.root)
        self.page_model_run = ctk.CTkFrame(self.root) # Renamed from page_model_select
        self.page_model_results = ctk.CTkFrame(self.root)
        # self.visualize_data = ctk.CTkFrame(self.root) # Not used currently

        self.create_upload_page()
        self.create_data_view_page()
        self.create_column_selection_page()
        self.create_results_page()
        self.create_model_run_page() # Renamed from create_model_selection_page
        self.create_model_results_display_page() # Renamed from create_model_results_page

    def create_upload_page(self):
        """Page 1: Upload CSV File"""
        # (No changes needed here)
        for widget in self.page_upload.winfo_children():
            widget.destroy()
        self.page_upload.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_upload,
            text="Welcome to ModelXpert.\nA complete ML solution for your needs.\n\n\nUpload CSV File",
            text_color='#E1FFBB',
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=30, padx=20)

        upload_btn = ctk.CTkButton(
            self.page_upload,
            text="Select File",
            command=self.load_csv,
            width=300,
            height=50,
            fg_color="#E2E2B6",
            font=("Helvetica", 24, "bold"),
            text_color='#001A6E',
            hover_color='#074799'
        )
        upload_btn.grid(row=1, column=0, pady=20)

        self.page_upload.pack(fill="both", expand=True)

    def create_data_view_page(self):
        """Page 2: Display Uploaded CSV Data"""
        # (No changes needed here except button text)
        for widget in self.page_data_view.winfo_children():
            widget.destroy()
        self.page_data_view.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_data_view,
            text="Uploaded CSV Data",
            font=("Helvetica", 32, "bold"),
            text_color="white"
        )
        title.grid(row=0, column=0, pady=20, padx=20)

        self.frame_table_original = ctk.CTkFrame(
            self.page_data_view, fg_color="#333333"
        )
        self.frame_table_original.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.page_data_view.rowconfigure(1, weight=1)

        next_btn = ctk.CTkButton(
            self.page_data_view,
            text="Next: Select Task, Label and Features", # Updated text
            command=self.show_page_column_select,
            width=300,
            fg_color="#1565C0",
            text_color="white",
            font=("Helvetica", 16)
        )
        next_btn.grid(row=2, column=0, pady=20)

    def create_column_selection_page(self):
        """Page 3: Select Task Type, Label, and Feature Columns"""
        for widget in self.page_column_select.winfo_children():
            widget.destroy()
        self.page_column_select.grid_columnconfigure(0, weight=1)
        # self.page_column_select.grid_rowconfigure(3, weight=1) # Configure row for features frame

        title = ctk.CTkLabel(
            self.page_column_select,
            text="Select Task Type, Label and Features", # Updated text
            font=("Helvetica", 32, "bold"),
            text_color="white"
        )
        title.grid(row=0, column=0, pady=10, padx=20, columnspan=2)

        # --- Task Type Selection ---
        task_frame = ctk.CTkFrame(self.page_column_select, fg_color="#333333")
        task_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        task_label = ctk.CTkLabel(
            task_frame, text="Select Task Type:", font=("Arial", 18, "bold"), text_color="white"
        )
        task_label.pack(side="left", padx=10, pady=5)

        ctk.CTkRadioButton(
            task_frame, text="Classification", variable=self.task_type, value="Classification",
            font=("Arial", 16), command=self.update_ui_for_task
        ).pack(side="left", padx=10, pady=5)
        ctk.CTkRadioButton(
            task_frame, text="Regression", variable=self.task_type, value="Regression",
            font=("Arial", 16), command=self.update_ui_for_task
        ).pack(side="left", padx=10, pady=5)
        # --- End Task Type Selection ---


        # --- Label Selection ---
        self.label_frame = ctk.CTkFrame(self.page_column_select, fg_color="#333333")
        self.label_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.label_frame.grid_columnconfigure(1, weight=1) # Make combobox expand

        label_label = ctk.CTkLabel(
            self.label_frame,
            text="Select Target (Label) Column:", # Clarified text
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        label_label.grid(row=0, column=0, pady=10, padx=10, sticky="w")

        # Use CTkComboBox for consistency
        self.label_menu = ctk.CTkComboBox(
            self.label_frame,
            state="readonly",
            variable=self.label_var,
            font=("Arial", 16), # Slightly larger font
            values=[], # Will be populated later
            command=self.update_feature_selection # Use command instead of bind
        )
        self.label_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        # --- End Label Selection ---


        # --- Feature Selection ---
        feature_title_label = ctk.CTkLabel(
            self.page_column_select,
            text="Select Feature Columns:",
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        feature_title_label.grid(row=3, column=0, padx=20, pady=(10,0), sticky="w")

        self.column_frame = ctk.CTkScrollableFrame(self.page_column_select, fg_color="#333333", label_text="") # Use ScrollableFrame
        self.column_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=5, sticky="nsew")
        self.page_column_select.rowconfigure(4, weight=1) # Make feature frame expand
        self.column_frame.grid_columnconfigure(0, weight=1)
        # --- End Feature Selection ---


        # --- Process Button ---
        self.process_btn = ctk.CTkButton(
            self.page_column_select,
            text="Process Data",
            command=self.process_data,
            width=300,
            fg_color="#1565C0",
            text_color="white",
            font=("Helvetica", 16)
        )
        self.process_btn.grid(row=5, column=0, columnspan=2, pady=20)
        # --- End Process Button ---


    def create_results_page(self):
        """Page 4: Show Processed & Original Data with Navigation Options"""
        for widget in self.page_results.winfo_children():
            widget.destroy()
        self.page_results.grid_columnconfigure(0, weight=1)

        nav_frame = ctk.CTkFrame(self.page_results)
        nav_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        nav_frame.grid_columnconfigure((0, 1, 2), weight=1) # Distribute space

        self.btn_original = ctk.CTkButton(
            nav_frame,
            text="View Original Data",
            command=lambda: self.show_table(self.df_original, "Original Data"),
            width=200,
            state="disabled",
            font=("Helvetica", 14)
        )
        self.btn_original.grid(row=0, column=0, padx=10, pady=5)

        self.btn_processed = ctk.CTkButton(
            nav_frame,
            text="View Processed Data",
            command=lambda: self.show_table(self.df_processed, "Processed Data"),
            width=200,
            state="disabled",
            font=("Helvetica", 14)
        )
        self.btn_processed.grid(row=0, column=1, padx=10, pady=5)

        # Button text will be updated based on task type
        self.btn_model_run_nav = ctk.CTkButton(
            nav_frame,
            text="Run Models", # Generic text initially
            command=self.go_to_model_run_page, # Use intermediate function
            width=250,
            state="disabled",
            font=("Helvetica", 14)
        )
        self.btn_model_run_nav.grid(row=0, column=2, padx=10, pady=5)

        # --- Table Display Area ---
        self.result_table_frame = ctk.CTkFrame(self.page_results, fg_color="#2b2b2b") # Slightly different bg
        self.result_table_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.page_results.rowconfigure(1, weight=1) # Make table frame expand

    def create_model_run_page(self):
        """Page 5: Model Processing Page (Handles both Classification & Regression)"""
        for widget in self.page_model_run.winfo_children():
            widget.destroy()
        self.page_model_run.grid_columnconfigure(0, weight=1)

        self.model_run_title = ctk.CTkLabel(
            self.page_model_run,
            text="Run Models", # Generic title initially
            font=("Helvetica", 32, "bold")
        )
        self.model_run_title.grid(row=0, column=0, pady=20, padx=20)

        self.model_run_info = ctk.CTkLabel(
            self.page_model_run,
            text="Click the button below to train models on the processed data.\nResults will be generated.", # Generic info
            font=("Helvetica", 16),
            justify="center"
        )
        self.model_run_info.grid(row=1, column=0, pady=10, padx=20)

        self.run_model_btn = ctk.CTkButton(
            self.page_model_run,
            text="Run Models", # Generic text initially
            command=self.run_models, # Updated command
            width=300,
            fg_color="#2196F3",
            font=("Helvetica", 16)
        )
        self.run_model_btn.grid(row=2, column=0, pady=20)

        back_btn = ctk.CTkButton(
            self.page_model_run,
            text="Back to Results Overview",
            command=lambda: self.show_page(self.page_results),
            width=300,
            font=("Helvetica", 16)
        )
        back_btn.grid(row=3, column=0, pady=10)

    def create_model_results_display_page(self):
        """Page 6: Display Model Results and Allow Model Dumping"""
        for widget in self.page_model_results.winfo_children():
            widget.destroy()
        self.page_model_results.grid_columnconfigure(0, weight=1)

        self.model_results_title = ctk.CTkLabel(
            self.page_model_results,
            text="Model Results and Dumping", # Generic title initially
            font=("Helvetica", 32, "bold")
        )
        self.model_results_title.grid(row=0, column=0, pady=20, padx=20)

        # Frame for displaying the model results DataFrame
        self.model_results_table_frame = ctk.CTkFrame(self.page_model_results, fg_color="#2b2b2b")
        self.model_results_table_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.page_model_results.rowconfigure(1, weight=1) # Make table frame expand

        # Button to dump selected models
        self.dump_btn = ctk.CTkButton(
            self.page_model_results,
            text="Dump Selected Models",
            command=self.dump_selected_models,
            width=300,
            fg_color="#2196F3",
            font=("Helvetica", 16),
            state="disabled" # Disable initially until results are shown
        )
        self.dump_btn.grid(row=2, column=0, pady=10)

        # Back button
        back_btn = ctk.CTkButton(
            self.page_model_results,
            text="Back to Model Run", # Updated text
            command=lambda: self.show_page(self.page_model_run),
            width=300,
            font=("Helvetica", 16)
        )
        back_btn.grid(row=3, column=0, pady=10)

    # --- Helper Functions for Page Navigation Safety ---
    def safe_show_page(self, page_func):
         if self.df_original is None:
              messagebox.showwarning("No Data", "Please load a CSV file first.")
              return
         page_func()

    def safe_show_page_data_view(self):
        self.safe_show_page(lambda: self.show_page(self.page_data_view))

    def safe_show_page_column_select(self):
        self.safe_show_page(self.show_page_column_select) # Calls the method that populates

    def safe_show_page_results(self):
        if self.df_processed is None:
             messagebox.showwarning("No Processed Data", "Please process the data first (Select Columns -> Process Data).")
             return
        self.show_page(self.page_results)
        # Show processed data by default when navigating here
        self.show_table(self.df_processed, "Processed Data")


    # --- Core Logic Methods ---

    def generate_ydata_report(self):
        """Generate a YData profiling report and open it in the browser."""
        if self.df_original is None:
            messagebox.showerror("Error", "No CSV file loaded!")
            return

        # Extract file name without extension
        try:
            report_file = os.path.join(self.save_dir, f"{self.file_name}_eda_report.html")

            # Generate the report
            profile = ProfileReport(self.df_original, title="Exploratory Data Analysis Report", explorative=True)
            profile.to_file(report_file)

            # Open the report in the default web browser
            webbrowser.open(f"file://{os.path.abspath(report_file)}") # Use file:// protocol for local files
            messagebox.showinfo("Success", f"EDA Report generated and opened:\n{report_file}")
        except AttributeError:
             messagebox.showerror("Error", "File path attribute not found. Please reload the CSV.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate report: {e}")

    def show_page(self, page):
        """Hide all pages and display the selected page."""
        for p in [
            self.page_upload, self.page_data_view, self.page_column_select,
            self.page_results, self.page_model_run, self.page_model_results
        ]:
            # Use pack_forget() instead of grid_forget()
            p.pack_forget()

        # Use pack() instead of grid() for the page frame
        # Add padding and make it fill the available space
        page.pack(fill="both", expand=True, padx=10, pady=10)

        # Remove grid configuration for the root window itself,
        # as its direct children are now managed by pack.
        # self.root.grid_rowconfigure(0, weight=1) # REMOVE or COMMENT OUT
        # self.root.grid_columnconfigure(0, weight=1) # REMOVE or COMMENT OUT


    def load_csv(self):
        """Load CSV file and display its contents."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.df_original = pd.read_csv(file_path)
            self.file_path = file_path # Store file path
            self.file_name = os.path.splitext(os.path.basename(self.file_path))[0]
            self.main_dir = os.getcwd()
            self.save_dir = os.path.join(self.main_dir, self.file_name)
            os.makedirs(self.save_dir, exist_ok=True)
            messagebox.showinfo("Success", "CSV file loaded successfully!")

            # Reset states when new file loaded
            self.df_processed = None
            self.global_model_results = None
            self.global_regression_results = None
            self.label_var.set("")
            self.feature_vars = []
            self.btn_original.configure(state="disabled")
            self.btn_processed.configure(state="disabled")
            self.btn_model_run_nav.configure(state="disabled")
            self.dump_btn.configure(state="disabled")


            # Display data and navigate
            self.display_table(self.df_original.head(100), self.frame_table_original, "Original Data (First 100 Rows)") # Show head
            self.show_page(self.page_data_view)

            # Update column selection page widgets
            self.label_menu.configure(values=list(self.df_original.columns))
            # Clear old feature checkboxes if any
            for widget in self.column_frame.winfo_children():
                widget.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Could not load file: {e}")
            self.df_original = None # Ensure df is None on error
            self.file_path = None


    def show_page_column_select(self, event=None):
        """Populate label and feature selection widgets (called when navigating to the page)."""
        if self.df_original is None:
             messagebox.showerror("Error", "Load a CSV file first.")
             return

        # Populate label dropdown
        current_label = self.label_var.get()
        self.label_menu.configure(values=list(self.df_original.columns))
        if current_label in self.df_original.columns:
            self.label_menu.set(current_label) # Keep selection if valid
        else:
             self.label_menu.set("") # Or clear if invalid/first time

        # Update feature checkboxes based on current label selection
        self.update_feature_selection()
        # Update UI elements based on task type
        self.update_ui_for_task()

        self.show_page(self.page_column_select)


    def update_feature_selection(self, event=None):
        """Update feature selection checkboxes based on chosen label."""
        if self.df_original is None: return # Should not happen if flow is correct

        label_column = self.label_var.get()

        # Clear existing checkboxes
        for widget in self.column_frame.winfo_children():
            widget.destroy()
        self.feature_vars = [] # Reset the list

        if not label_column: # Don't show features if no label is selected
            return

        remaining_columns = [col for col in self.df_original.columns if col != label_column]

        # Create new checkboxes inside the scrollable frame
        for idx, col in enumerate(remaining_columns):
            var = ctk.IntVar(value=1) # Default to selected
            chk = ctk.CTkCheckBox(
                self.column_frame, # Add to the scrollable frame
                text=col,
                variable=var,
                font=("Arial", 14), # Smaller font for checkbox list
                text_color="white"
            )
            chk.grid(row=idx, column=0, sticky="w", padx=10, pady=2) # Use grid within the frame
            self.feature_vars.append((col, var))


    def update_ui_for_task(self):
        """Update button texts and titles based on selected task type."""
        task = self.task_type.get()
        run_button_text = f"Run {task}"
        results_title_text = f"{task} Model Results"
        run_page_title = f"{task} Models"
        run_page_info = f"Click the button below to train {task.lower()} models on the processed data.\nResults will be generated."

        # Update Model Run Page
        self.model_run_title.configure(text=run_page_title)
        self.model_run_info.configure(text=run_page_info)
        self.run_model_btn.configure(text=run_button_text)

        # Update Results Page Navigation Button
        self.btn_model_run_nav.configure(text=run_button_text)

        # Update Model Results Display Page Title
        self.model_results_title.configure(text=results_title_text)


    def process_data(self):
        """Process selected label and features based on task type."""
        label_column = self.label_var.get()
        selected_features = [col for col, var in self.feature_vars if var.get() == 1]
        task = self.task_type.get()

        if not label_column:
            messagebox.showerror("Error", "Please select a target (label) column!")
            return
        if not selected_features:
            messagebox.showerror("Error", "Please select at least one feature column!")
            return

        # --- Task Specific Validation ---
        if task == "Regression":
            # Check if target column is numeric-like
            try:
                # Attempt conversion to numeric, ignore errors for now
                pd.to_numeric(self.df_original[label_column], errors='raise')
            except (ValueError, TypeError):
                messagebox.showerror("Error", f"For Regression, the target column ('{label_column}') must contain numeric data.")
                return
        # (Could add check for classification: target shouldn't have too many unique values)

        try:
            # Select only the necessary columns
            cols_to_keep = selected_features + [label_column]
            data_subset = self.df_original[cols_to_keep].copy()

            # Preprocess data (function now depends on task type)
            self.df_processed = self.preprocess_data(data_subset, label_column, task)

            if self.df_processed is not None:
                # Enable buttons on results page
                self.btn_original.configure(state="normal")
                self.btn_processed.configure(state="normal")
                self.btn_model_run_nav.configure(state="normal")

                # # Save preprocessed data
                # output_filename = f"preprocessed_data_{task.lower()}.csv"
                # self.df_processed.to_csv(output_filename, index=False)
                # messagebox.showinfo("Success", f"Data processed for {task}.\nPreprocessed data saved as '{output_filename}'."
                # Save preprocessed data inside that folder
                output_filename = os.path.join(self.save_dir, f"{self.file_name}_preprocessed_data_{task.lower()}.csv")
                self.df_processed.to_csv(output_filename, index=False)

                # Show success message
                messagebox.showinfo(
                    "Success",
                    f"Data processed for {task}.\nPreprocessed data saved at:\n'{output_filename}'"
                )

                # Navigate to results page and show processed data
                self.show_page(self.page_results)
                self.show_table(self.df_processed.head(100), "Processed Data (First 100 Rows)")

        except Exception as e:
            messagebox.showerror("Error", f"Error during data processing: {e}")
            self.df_processed = None # Reset on error
            # Disable buttons if processing failed
            self.btn_processed.configure(state="disabled")
            self.btn_model_run_nav.configure(state="disabled")


    def preprocess_data(self, data, label_col, task_type):
        """Preprocessing: fill missing, scale numerics (excluding target), encode categoricals (excluding target)."""
        try:
            # Make a copy to avoid SettingWithCopyWarning
            df_processed = data.copy()

            # --- Handle Missing Values ---
            numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
            # Fill numeric NaNs (excluding target column)
            for col in numeric_cols:
                if col == label_col: continue # Skip target
                if df_processed[col].isnull().any():
                    non_missing = df_processed[col].dropna()
                    # Use mean for simplicity, GMean needs positive values
                    fill_value = non_missing.mean() if not non_missing.empty else 0
                    # Check if fill_value is NaN (e.g., column was all NaNs)
                    if pd.isna(fill_value): fill_value = 0
                    df_processed[col].fillna(fill_value, inplace=True)

            # Fill categorical NaNs with mode or 'Missing'
            categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()
            for col in categorical_cols:
                 if col == label_col and task_type == 'Regression': continue # Skip object target in regression
                 if df_processed[col].isnull().any():
                    mode_val = df_processed[col].mode()
                    fill_value = mode_val[0] if not mode_val.empty else 'Missing'
                    df_processed[col].fillna(fill_value, inplace=True)

            # Drop rows if target variable is missing (essential)
            df_processed.dropna(subset=[label_col], inplace=True)
            if df_processed.empty:
                messagebox.showerror("Error", "No data remaining after dropping rows with missing target values.")
                return None


            # --- Scaling Numeric Features ---
            features_to_scale = [col for col in numeric_cols if col != label_col]
            if features_to_scale: # Only scale if there are numeric features
                scaler = StandardScaler()
                df_processed[features_to_scale] = scaler.fit_transform(df_processed[features_to_scale])
                save_dir = os.path.join(os.path.dirname(self.save_dir), self.file_name)

                # Save the scaler in that folder
                scaler_file = os.path.join(self.save_dir, f"scaler_{task_type.lower()}.pkl")
                joblib.dump(scaler, scaler_file)
                print(f"Scaler saved at {scaler_file}")


            # --- Encoding Categorical Features ---
            label_encoders = {}
            features_to_encode = [col for col in categorical_cols if col != label_col]
            for col in features_to_encode:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                label_encoders[col] = le

            # --- Encode Target Variable (Only for Classification) ---
            if task_type == 'Classification' and label_col in categorical_cols:
                target_le = LabelEncoder()
                df_processed[label_col] = target_le.fit_transform(df_processed[label_col])
                label_encoders[label_col] = target_le # Store target encoder too

                joblib.dump(target_le, "target_encoder.pkl") # Save target encoder separately

            # Now save the encoders in that folder
            if label_encoders:  # Check if dictionary is not empty
                encoder_file = os.path.join(self.save_dir, f"encoders_{task_type.lower()}.pkl")
                joblib.dump(label_encoders, encoder_file)
                print(f"Encoders saved at {encoder_file}")

            return df_processed

        except Exception as e:
            messagebox.showerror("Error", f"Error during preprocessing: {e}")
            return None


    def display_table(self, df, frame, title):
        """Display a pandas DataFrame in a tkinter Treeview within the specified frame."""
        # Clear previous widgets in the frame
        for widget in frame.winfo_children():
            widget.destroy()

        if df is None or df.empty:
            ctk.CTkLabel(frame, text=f"{title}\n(No data to display)", font=("Arial", 16)).pack(pady=20)
            return

        title_label = ctk.CTkLabel(frame, text=title, font=("Arial", 18, "bold"), text_color="white")
        title_label.pack(pady=(10, 5), anchor="n")

        # Use CTkFrame for better integration
        container = ctk.CTkFrame(frame, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=5, pady=5)

        # Configure Treeview Style
        style = ttk.Style()
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#333333", borderwidth=0)
        style.map('Treeview', background=[('selected', '#0078D7')]) # Selection color
        style.configure("Treeview.Heading", background="#3c3c3c", foreground="cyan", font=('Arial', 11, 'bold'), borderwidth=0)
        style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nsew'})]) # Remove borders

        # Create Treeview
        tree = tk_tt.Treeview(container, columns=list(df.columns), show="headings", style="Treeview")

        # Scrollbars (using CTkScrollbar if preferred, but ttk works fine)
        vsb = tk_tt.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = tk_tt.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid layout for Treeview and Scrollbars
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Define column headings and properties
        for col in df.columns:
            tree.heading(col, text=col, anchor='w') # Anchor West for better readability
            # Estimate column width (simple heuristic)
            try:
                max_len = max(df[col].astype(str).map(len).max(), len(col))
                col_width = max(60, min(250, max_len * 9)) # Adjust multiplier as needed
            except:
                 col_width = 100 # Default
            tree.column(col, anchor="w", width=col_width, stretch=False) # Don't stretch initially

        # Insert data rows
        for index, row in df.iterrows():
            # Convert all values to string for display, handle potential NaNs
            values = [str(v) if pd.notna(v) else "" for v in row.tolist()]
            tree.insert("", "end", values=values, iid=index) # Use index as iid

    def show_table(self, df, title):
        """Show a data table in the Results page's dedicated frame."""
        self.display_table(df, self.result_table_frame, title)

    def go_to_model_run_page(self):
        """Navigate to the model running page."""
        self.update_ui_for_task() # Ensure texts are correct before showing
        self.show_page(self.page_model_run)


    def run_models(self):
        """Run appropriate models (Classification or Regression) and display results."""
        if self.df_processed is None:
            messagebox.showerror("Error", "No processed data available. Please process data first.")
            return

        label_column = self.label_var.get()
        task = self.task_type.get()

        if label_column not in self.df_processed.columns:
            messagebox.showerror("Error", f"Target column '{label_column}' not found in processed data.")
            return

        try:
            X = self.df_processed.drop(columns=[label_column])
            y = self.df_processed[label_column]

            # Check if X or y is empty after preprocessing/filtering
            if X.empty or y.empty:
                 messagebox.showerror("Error", "No data left for modeling after preprocessing.")
                 return

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Reset global results before running new models
            fcp.global_model_results = pd.DataFrame(columns=fcp.global_model_results.columns)
            fcp.global_model_results2 = pd.DataFrame(columns=fcp.global_model_results2.columns)


            # --- Run models based on task type ---
            if task == "Classification":
                print(f"Running Classification Models for target: {label_column}")
                # Add more classification models as needed from final_pipeline
                fcp.train_decision_tree(X_train, y_train, X_test, y_test)
                fcp.train_random_forest(X_train, y_train, X_test, y_test)
                fcp.train_knn(X_train, y_train, X_test, y_test)
                fcp.train_svc(X_train, y_train, X_test, y_test)
                fcp.train_logistic_regression(X_train, y_train, X_test, y_test)
                self.global_model_results = fcp.global_model_results # Store results
                results_df = self.global_model_results
                print("Classification Results:")
                print(results_df)


            elif task == "Regression":
                print(f"Running Regression Models for target: {label_column}")
                # Add more regression models as needed from final_pipeline
                fcp.train_svr(X_train, y_train, X_test, y_test)
                fcp.train_linear_regression(X_train, y_train, X_test, y_test)
                # Assuming train_random_forest_regressor exists in final_pipeline:
                # fcp.train_random_forest_regressor(X_train, y_train, X_test, y_test)
                self.global_regression_results2 = fcp.global_model_results2 # Store results
                results_df = self.global_regression_results2
                print("Regression Results:")
                print(results_df)

            else:
                messagebox.showerror("Error", f"Unknown task type: {task}")
                return

            # --- Display Results ---
            if results_df is not None and not results_df.empty:
                self.display_results_table(results_df, task)
                self.dump_btn.configure(state="normal") # Enable dump button
                self.show_page(self.page_model_results) # Navigate to results display
            else:
                messagebox.showinfo("Info", f"No {task.lower()} models were successfully trained or no results generated.")
                # Stay on the run page or go back? Go back for now.
                self.show_page(self.page_model_run)


        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during model training or evaluation: {e}")
            print(f"Traceback: {e.__traceback__}") # More debug info

    def display_results_table(self, results_df, task_type):
        # Clear previous content
        for widget in self.model_results_table_frame.winfo_children():
            widget.destroy()

        if results_df is None or results_df.empty:
            ctk.CTkLabel(self.model_results_table_frame, text="No model results to display.", font=("Arial", 16)).pack(
                pady=20)
            self.dump_btn.configure(state="disabled")  # Disable dump if no results
            return

        # Determine columns based on task type (keep this part)
        if task_type == "Classification":
            display_columns = ['model_name', 'algorithm', 'accuracy', 'f1_score']
            results_df_display = results_df[display_columns].copy()
            results_df_display['accuracy'] = results_df_display['accuracy'].map('{:.4f}'.format)
            results_df_display['f1_score'] = results_df_display['f1_score'].map('{:.4f}'.format)
        elif task_type == "Regression":
            display_columns = ['model_name', 'algorithm', 'r2_score', 'rmse', 'mae']
            results_df_display = results_df[display_columns].copy()
            results_df_display['r2_score'] = results_df_display['r2_score'].map('{:.4f}'.format)
            results_df_display['rmse'] = results_df_display['rmse'].map('{:.2f}'.format)
            results_df_display['mae'] = results_df_display['mae'].map('{:.2f}'.format)
        else:
            messagebox.showerror("Internal Error", f"Cannot display results for unknown task: {task_type}")
            return

        # Add '#' column for row numbers (still useful reference)
        results_df_display.insert(0, '#', range(1, len(results_df_display) + 1))

        # --- Create Treeview ---
        container = ctk.CTkFrame(self.model_results_table_frame, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=5, pady=5)

        style = ttk.Style()
        style.configure("Results.Treeview", background="#2b2b2b", foreground="white", fieldbackground="#333333",
                        rowheight=25)
        style.map('Results.Treeview', background=[('selected', '#0078D7')])  # Highlight selected rows
        style.configure("Results.Treeview.Heading", background="#3c3c3c", foreground="cyan", font=('Arial', 11, 'bold'))

        # *** Key Change: Set selectmode to 'extended' ***
        tree = tk_tt.Treeview(container, columns=list(results_df_display.columns), show="headings",
                              style="Results.Treeview", selectmode='extended')

        # Store tree reference for later use in dump_selected_models
        self.results_treeview = tree  # Add this line

        vsb = tk_tt.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = tk_tt.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Define column headings and widths
        tree.column("#", width=40, anchor='center', stretch=False)
        tree.heading("#", text="#", anchor='center')
        # --- Removed 'Select' column ---

        for col in display_columns:  # The actual metric columns
            tree.heading(col, text=col, anchor='center')
            width = 180 if col == 'model_name' else (100 if col == 'algorithm' else 90)
            tree.column(col, anchor="center", width=width, stretch=True)

        # --- Add Data ---
        # --- Removed checkbox creation and placement ---
        # --- Removed self.model_selection_vars and self.checkbuttons ---

        for i, (idx, row_data) in enumerate(results_df_display.iterrows()):
            # `idx` is the original index from results_df
            # *** Key Change: Use original DataFrame index (idx) as item ID (iid) ***
            tree.insert("", "end", values=row_data.tolist(), iid=idx)

        # Enable dump button
        self.dump_btn.configure(state="normal")

        # Optional: Add a small instruction label
        instruction_label = ctk.CTkLabel(self.model_results_table_frame,
                                         text="Use Ctrl+Click or Shift+Click in the table to select multiple models.",
                                         font=("Arial", 10), text_color="grey")
        instruction_label.pack(pady=(5, 0), side="bottom", anchor="w", padx=5)

    def dump_selected_models(self):
        """Dump selected models (using Treeview selection) to .pkl files."""
        task = self.task_type.get()

        print(f"\n--- Starting Model Dump Process ---")
        print(f"Current Task Type: {task}")
        print(f"Current Working Directory: {os.getcwd()}")

        # Get selection from the Treeview
        if not hasattr(self, 'results_treeview'):
            messagebox.showerror("Error", "Results table not found.")
            return

        selected_iids_str = self.results_treeview.selection()

        if not selected_iids_str:
            messagebox.showwarning("No Selection", "Please select at least one model from the table to dump.")
            return

        # Convert string iids to integers
        try:
            selected_indices = [int(iid) for iid in selected_iids_str]
        except ValueError:
            messagebox.showerror("Error", "Invalid selection ID encountered.")
            return

        # Get the correct results DataFrame based on task type (Corrected Check)
        results_df = None
        if task == "Classification":
            results_df = self.global_model_results
            if results_df is None:
                messagebox.showerror("Error", "Classification results not found (Internal Error: Results are None).")
                return
            if results_df.empty:
                messagebox.showerror("Error", "Classification results are empty. Cannot dump.")
                return
        elif task == "Regression":
            results_df = self.global_regression_results2
            if results_df is None:
                messagebox.showerror("Error", "Regression results not found (Internal Error: Results are None).")
                return
            if results_df.empty:
                messagebox.showerror("Error", "Regression results are empty. Cannot dump.")
                return
        else:
            messagebox.showerror("Internal Error", f"Unknown task type: {task}")
            return

        # --- Prepare Confirmation Dialog ---
        # --- Define models_to_dump_info BEFORE using it in the message ---
        confirm_message = f"You have selected {len(selected_indices)} model(s) for dumping:\n\n"
        models_to_dump_info = []  # <<< Define the list HERE

        for idx in selected_indices:
            try:
                model_info = results_df.loc[idx]
                model_name = model_info["model_name"]
                algorithm = model_info["algorithm"]
                confirm_message += f"- {model_name} ({algorithm})\n"
                models_to_dump_info.append(model_info)  # <<< Populate the list HERE
            except KeyError:
                messagebox.showerror("Error", f"Could not find model information for index {idx}.")
                return

        confirm_message += "\nDo you want to proceed with dumping these models?"

        # --- Show Confirmation Dialog and Define confirm ---
        confirm = messagebox.askyesno("Confirm Model Dump", confirm_message)  # <<< Define confirm HERE

        # --- Now check confirm AFTER it's defined ---
        if not confirm:
            print("Model dump cancelled by user.")  # Now this print is safe
            return

        # --- Proceed with Dumping ---
        print("User confirmed dump.")  # Add a print here if desired

        dump_count = 0
        dump_errors = 0
        model_dir = f"dumped_{task.lower()}_models"
        absolute_model_dir = self.save_dir

        print(f"Attempting to create/use directory: {absolute_model_dir}")

        # Create the directory if it doesn't exist
        try:
            os.makedirs(absolute_model_dir, exist_ok=True)
            print(f"Directory confirmed/created: {absolute_model_dir}")
        except OSError as e:
            messagebox.showerror("Directory Error", f"Could not create directory: {absolute_model_dir}\nError: {e}")
            print(f"Error creating directory: {e}")
            return

        try:
            # Prepare full dataset for final training
            label_column = self.label_var.get()
            if self.df_processed is None or label_column not in self.df_processed.columns:
                raise ValueError("Processed data or label column is missing for final model training.")
            X_full = self.df_processed.drop(columns=[label_column])
            y_full = self.df_processed[label_column]

            print(f"Prepared full dataset for retraining. X shape: {X_full.shape}, y shape: {y_full.shape}")

            # --- Now iterate through models_to_dump_info AFTER it's defined and populated ---
            for model_info in models_to_dump_info:
                model_name = model_info["model_name"]
                algorithm = model_info["algorithm"]
                params = model_info["hyperparameters"]
                base_filename = model_info["model_file"]
                model_filename = os.path.join(absolute_model_dir, base_filename)

                print(f"\nProcessing model: {model_name} ({algorithm})")
                print(f"  Parameters: {params}")
                print(f"  Attempting to save to: {model_filename}")

                try:
                    # Initialize the correct model class
                    model = None
                    if task == "Classification":
                        if algorithm == "RandomForest":
                            model = RandomForestClassifier(**params, random_state=42)
                        elif algorithm == "DecisionTree":
                            model = DecisionTreeClassifier(**params, random_state=42)
                        elif algorithm == "KNN":
                            model = KNeighborsClassifier(**params)
                        elif algorithm == "SVC":
                            model = SVC(**params, probability=True)
                        elif algorithm == "Logistic Regression":
                            model = LogisticRegression(**params, random_state=42)
                    elif task == "Regression":
                        if algorithm == "SVR":
                            model = SVR(**params)
                        elif algorithm == "Linear Regression":
                            model = Ridge(**params)
                        elif algorithm == "RandomForest":
                            model = RandomForestRegressor(**params, random_state=42)

                    if model is None:
                        print(f"  Warning: Skipping unsupported algorithm '{algorithm}' for task '{task}'.")
                        dump_errors += 1
                        continue

                    # Re-train the model
                    print(f"  Retraining final model: {model_name}...")
                    model.fit(X_full, y_full)
                    print(f"  Retraining complete.")

                    # Save the trained model
                    print(f"  Attempting joblib.dump()...")
                    joblib.dump(model, model_filename)
                    print(f"  Successfully dumped: {model_filename}")
                    dump_count += 1

                except Exception as train_dump_e:
                    print(f"  ERROR training/dumping model {model_name}: {train_dump_e}")
                    # import traceback
                    # print(traceback.format_exc())
                    dump_errors += 1

            # --- Final Message ---
            print(f"\n--- Dump Process Finished ---")
            print(f"Successful dumps: {dump_count}, Errors: {dump_errors}")
            success_msg = f"Successfully dumped {dump_count} model(s) to '{absolute_model_dir}'."
            error_msg = f" Failed to dump {dump_errors} model(s)." if dump_errors > 0 else ""
            messagebox.showinfo("Dump Complete", success_msg + error_msg)

        except ValueError as ve:
            messagebox.showerror("Data Error", f"Could not prepare data for final model training:\n{ve}")
            print(f"Data preparation error: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during the model dumping process: {e}")
            print(f"Unexpected error during dumping: {e}")
            # import traceback
            # print(traceback.format_exc())


if __name__ == "__main__":
    # Ensure the root window is created correctly
    root = ctk.CTk()

    # Set main window properties *after* creating it
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    app = CSVProcessorApp(root)
    root.mainloop()