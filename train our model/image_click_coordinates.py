#!/usr/bin/env python3
"""
Interactive image click coordinate recorder with two sessions:
1. Galaxy members (cluster members) - Red markers
2. Background objects (cluster non-members) - Blue markers

Keyboard shortcuts:
- Press 'M' to switch to Galaxy Members mode
- Press 'B' to switch to Background Objects mode
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import csv
import os
import sys
from datetime import datetime
import shutil
import re


class ImageClickRecorder:
    def __init__(self, root, image_path):
        self.root = root
        self.image_path = image_path
        self.members_coordinates = []
        self.non_members_coordinates = []
        self.current_mode = "members"  # "members" or "non_members"
        self.output_folder = None
        self.member_markers = []  # Store canvas markers for redrawing
        self.non_member_markers = []
        
        # Zoom variables
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.zoom_step = 0.1
        
        # Setup output folder
        self.setup_output_folder()
        
        # Copy original image to output folder
        self.save_original_image()
        
        # Add UI controls FIRST so they reserve space on the right (visible)
        self.add_controls()
        
        # Load and display image (canvas goes on the left)
        self.load_image()
        
        # Setup CSV files
        self.setup_csv()
        
        # Bind click event
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Bind zoom events
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows/Mac
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())  # Focus on mouse enter
        
        # Bind keyboard shortcuts for mode switching
        self.root.bind("<KeyPress-m>", lambda e: self.switch_mode("members"))
        self.root.bind("<KeyPress-M>", lambda e: self.switch_mode("members"))
        self.root.bind("<KeyPress-b>", lambda e: self.switch_mode("non_members"))
        self.root.bind("<KeyPress-B>", lambda e: self.switch_mode("non_members"))
        self.root.focus_set()  # Allow root window to receive keyboard events
        
    def setup_output_folder(self):
        """Create output folder with cluster serial number"""
        # Use current working directory where script is run
        base_dir = os.getcwd()
        
        # Find all existing cluster folders
        cluster_numbers = []
        pattern = re.compile(r'^cluster_(\d{3})$')
        
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    match = pattern.match(item)
                    if match:
                        cluster_numbers.append(int(match.group(1)))
        
        # Determine next cluster number
        if cluster_numbers:
            next_number = max(cluster_numbers) + 1
        else:
            next_number = 1
        
        # Format as 3-digit number
        cluster_number_str = f"{next_number:03d}"
        self.output_folder = os.path.join(base_dir, f"cluster_{cluster_number_str}")
        
        # Create folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output folder created: {self.output_folder}\n")
    
    def save_original_image(self):
        """Copy original image to output folder as 'image'"""
        output_image_path = os.path.join(self.output_folder, "image.png")
        shutil.copy2(self.image_path, output_image_path)
        print(f"Original image saved to: {output_image_path}")
    
    def load_image(self):
        """Load and display the image"""
        try:
            # Open image with PIL
            self.pil_image = Image.open(self.image_path).copy()
            self.original_image = self.pil_image.copy()  # Keep original for saving
            
            # Get image dimensions
            self.original_width, self.original_height = self.pil_image.size
            
            # Create frame with scrollbars for zooming
            canvas_frame = tk.Frame(self.root)
            canvas_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
            
            # Create scrollbars
            self.v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
            self.h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
            self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Create canvas with scrollbars
            self.canvas = tk.Canvas(
                canvas_frame,
                width=self.original_width,
                height=self.original_height,
                cursor="crosshair",
                yscrollcommand=self.v_scrollbar.set,
                xscrollcommand=self.h_scrollbar.set
            )
            self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.v_scrollbar.config(command=self.canvas.yview)
            self.h_scrollbar.config(command=self.canvas.xview)
            
            # Convert PIL image to PhotoImage for tkinter
            self.update_display()
            
            # Set window size - reserve space for control panel (280px) so mode buttons stay visible
            panel_width = 280
            max_width = min(self.original_width + panel_width, 1400)
            max_height = min(self.original_height + 150, 900)
            self.root.geometry(f"{max_width}x{max_height}")
            self.root.title(f"Image Click Recorder - {os.path.basename(self.image_path)}")
            self.root.minsize(600, 400)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            sys.exit(1)
    
    def update_display(self):
        """Update the displayed image with current zoom level"""
        # Calculate new size based on zoom
        new_width = int(self.original_width * self.zoom_level)
        new_height = int(self.original_height * self.zoom_level)
        
        # Resize image
        resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Update canvas scroll region
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Draw image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Redraw all markers (scale coordinates by zoom level)
        marker_size = 2  # Smaller markers
        for x, y in self.member_markers:
            scaled_x = x * self.zoom_level
            scaled_y = y * self.zoom_level
            self.canvas.create_oval(
                scaled_x - marker_size, scaled_y - marker_size,
                scaled_x + marker_size, scaled_y + marker_size,
                fill='red', outline='darkred', width=1
            )
        
        for x, y in self.non_member_markers:
            scaled_x = x * self.zoom_level
            scaled_y = y * self.zoom_level
            self.canvas.create_oval(
                scaled_x - marker_size, scaled_y - marker_size,
                scaled_x + marker_size, scaled_y + marker_size,
                fill='blue', outline='darkblue', width=1
            )
    
    def setup_csv(self):
        """Setup CSV files for saving coordinates"""
        # Create CSV files with headers
        members_csv = os.path.join(self.output_folder, "cluster_members.csv")
        non_members_csv = os.path.join(self.output_folder, "cluster_non_members.csv")
        
        with open(members_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y'])
        
        with open(non_members_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y'])
        
        print("CSV files created:")
        print(f"  - {members_csv}")
        print(f"  - {non_members_csv}\n")
    
    def on_click(self, event):
        """Handle mouse click on image"""
        # Get canvas coordinates (accounting for scroll position)
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to original image coordinates (divide by zoom level)
        x = int(canvas_x / self.zoom_level)
        y = int(canvas_y / self.zoom_level)
        
        # Clamp coordinates to image bounds
        x = max(0, min(x, self.original_width - 1))
        y = max(0, min(y, self.original_height - 1))
        
        marker_size = 2  # Smaller markers
        
        if self.current_mode == "members":
            self.members_coordinates.append((x, y))
            self.member_markers.append((x, y))
            color = 'red'
            outline_color = 'darkred'
            mode_name = "Galaxy Members"
        else:
            self.non_members_coordinates.append((x, y))
            self.non_member_markers.append((x, y))
            color = 'blue'
            outline_color = 'darkblue'
            mode_name = "Background Objects"
        
        # Print to console
        print(f"[{mode_name}] Clicked at: X={x}, Y={y}")
        
        # Save to CSV immediately
        self.save_coordinate_to_csv(x, y)
        
        # Visual feedback: draw a small circle at click point (use canvas coordinates)
        self.canvas.create_oval(
            canvas_x - marker_size, canvas_y - marker_size,
            canvas_x + marker_size, canvas_y + marker_size,
            fill=color, outline=outline_color, width=1
        )
    
    def save_coordinate_to_csv(self, x, y):
        """Save coordinate to appropriate CSV file"""
        if self.current_mode == "members":
            csv_path = os.path.join(self.output_folder, "cluster_members.csv")
        else:
            csv_path = os.path.join(self.output_folder, "cluster_non_members.csv")
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([x, y])
    
    def save_marked_image(self, mode):
        """Save marked image for the specified mode"""
        # Create a copy of the original image and convert to RGB if needed
        marked_image = self.original_image.copy()
        
        # Convert to RGB mode if image is in a different mode (e.g., RGBA, P)
        if marked_image.mode != 'RGB':
            # Create a white background for transparency
            rgb_image = Image.new('RGB', marked_image.size, (255, 255, 255))
            if marked_image.mode == 'RGBA':
                rgb_image.paste(marked_image, mask=marked_image.split()[3])  # Use alpha channel as mask
            else:
                rgb_image.paste(marked_image)
            marked_image = rgb_image
        
        draw = ImageDraw.Draw(marked_image)
        marker_size = 2  # Smaller markers
        
        if mode == "members":
            coordinates = self.members_coordinates
            fill_color = (255, 0, 0)  # Red in RGB
            outline_color = (200, 0, 0)  # Dark red
            filename = "image_marked_members.png"
        else:
            coordinates = self.non_members_coordinates
            fill_color = (0, 0, 255)  # Blue in RGB
            outline_color = (0, 0, 200)  # Dark blue
            filename = "image_marked_non_members.png"
        
        # Draw markers on the image
        for x, y in coordinates:
            # Draw a circle with fill and outline
            draw.ellipse([x - marker_size, y - marker_size, x + marker_size, y + marker_size], 
                        fill=fill_color, outline=outline_color, width=1)
        
        # Save the marked image
        output_path = os.path.join(self.output_folder, filename)
        marked_image.save(output_path)
        print(f"Marked image saved: {output_path}")
    
    def switch_mode(self, new_mode):
        """Switch between members and non-members mode"""
        # Save current marked image before switching
        if self.current_mode == "members" and self.members_coordinates:
            self.save_marked_image("members")
        elif self.current_mode == "non_members" and self.non_members_coordinates:
            self.save_marked_image("non_members")
        
        self.current_mode = new_mode
        self.update_status()
        mode_name = "Galaxy Members" if new_mode == "members" else "Background Objects"
        print(f"\nSwitched to: {mode_name} mode")
    
    def finish_session(self):
        """Finish current session and save marked image"""
        if self.current_mode == "members":
            if self.members_coordinates:
                self.save_marked_image("members")
                print(f"\nGalaxy Members session complete: {len(self.members_coordinates)} points recorded")
        else:
            if self.non_members_coordinates:
                self.save_marked_image("non_members")
                print(f"\nBackground Objects session complete: {len(self.non_members_coordinates)} points recorded")
    
    def on_mousewheel(self, event):
        """Handle mouse wheel zoom"""
        # Windows uses delta, Linux uses num
        if hasattr(event, 'delta'):
            if event.delta > 0:  # Scroll up (Windows/Mac)
                self.zoom_in()
            elif event.delta < 0:  # Scroll down (Windows/Mac)
                self.zoom_out()
        elif event.num == 4:  # Linux scroll up
            self.zoom_in()
        elif event.num == 5:  # Linux scroll down
            self.zoom_out()
    
    def zoom_in(self):
        """Zoom in"""
        if self.zoom_level < self.max_zoom:
            self.zoom_level = min(self.zoom_level + self.zoom_step, self.max_zoom)
            self.update_display()
            self.update_zoom_label()
    
    def zoom_out(self):
        """Zoom out"""
        if self.zoom_level > self.min_zoom:
            self.zoom_level = max(self.zoom_level - self.zoom_step, self.min_zoom)
            self.update_display()
            self.update_zoom_label()
    
    def zoom_reset(self):
        """Reset zoom to 100%"""
        self.zoom_level = 1.0
        self.update_display()
        self.update_zoom_label()
    
    def add_controls(self):
        """Add control panel with buttons - packed first so it stays visible on the right"""
        control_frame = tk.Frame(self.root, bg="lightgray", relief=tk.RAISED, borderwidth=2, width=260)
        control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        control_frame.pack_propagate(False)  # Keep fixed width so panel is never squeezed
        
        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="Current Mode: Galaxy Members",
            font=("Arial", 12, "bold"),
            bg="lightcoral",
            padx=10,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2
        )
        self.status_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(
            control_frame,
            text="Click on the image to mark points.\nUse mouse wheel to zoom.\n\nSwitch modes:\n• Press 'M' for Members\n• Press 'B' for Background\n• Or use buttons below",
            font=("Arial", 9),
            justify=tk.LEFT,
            wraplength=200,
            bg="lightyellow",
            padx=5,
            pady=5,
            relief=tk.SUNKEN,
            borderwidth=1
        )
        instructions.pack(pady=10)
        
        # Zoom controls
        tk.Label(control_frame, text="Zoom:", font=("Arial", 10, "bold"), bg="lightgray").pack(pady=(10, 5))
        zoom_frame = tk.Frame(control_frame, bg="lightgray")
        zoom_frame.pack(pady=5)
        
        zoom_in_btn = tk.Button(
            zoom_frame,
            text="+",
            command=self.zoom_in,
            font=("Arial", 12, "bold"),
            width=3
        )
        zoom_in_btn.pack(side=tk.LEFT, padx=2)
        
        zoom_out_btn = tk.Button(
            zoom_frame,
            text="-",
            command=self.zoom_out,
            font=("Arial", 12, "bold"),
            width=3
        )
        zoom_out_btn.pack(side=tk.LEFT, padx=2)
        
        reset_btn = tk.Button(
            zoom_frame,
            text="Reset",
            command=self.zoom_reset,
            font=("Arial", 8),
            width=5
        )
        reset_btn.pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = tk.Label(
            control_frame,
            text="Zoom: 100%",
            font=("Arial", 9),
            bg="lightgray"
        )
        self.zoom_label.pack(pady=2)
        
        # Mode buttons
        mode_label = tk.Label(control_frame, text="Switch Mode:", font=("Arial", 10, "bold"), bg="lightgray")
        mode_label.pack(pady=(10, 5))
        
        members_btn = tk.Button(
            control_frame,
            text="Mark Galaxy Members (M)",
            command=lambda: self.switch_mode("members"),
            bg="lightcoral",
            font=("Arial", 10, "bold"),
            width=22,
            relief=tk.RAISED,
            borderwidth=3,
            activebackground="coral"
        )
        members_btn.pack(pady=5)
        
        non_members_btn = tk.Button(
            control_frame,
            text="Mark Background Objects (B)",
            command=lambda: self.switch_mode("non_members"),
            bg="lightblue",
            font=("Arial", 10, "bold"),
            width=22,
            relief=tk.RAISED,
            borderwidth=3,
            activebackground="skyblue"
        )
        non_members_btn.pack(pady=5)
        
        # Finish session button
        tk.Label(control_frame, text="", font=("Arial", 1), bg="lightgray").pack(pady=10)
        finish_btn = tk.Button(
            control_frame,
            text="Finish Current Session",
            command=self.finish_session,
            bg="lightgray",
            font=("Arial", 10),
            width=20
        )
        finish_btn.pack(pady=5)
        
        # Statistics
        self.stats_label = tk.Label(
            control_frame,
            text="Members: 0\nNon-members: 0",
            font=("Arial", 9),
            justify=tk.LEFT,
            bg="lightgray"
        )
        self.stats_label.pack(pady=10)
        
        # Update status
        self.update_status()
    
    def update_zoom_label(self):
        """Update zoom level label"""
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")
    
    def update_status(self):
        """Update status label and statistics"""
        if self.current_mode == "members":
            self.status_label.config(text="Current Mode: Galaxy Members", bg="lightcoral")
        else:
            self.status_label.config(text="Current Mode: Background Objects", bg="lightblue")
        
        self.stats_label.config(
            text=f"Members: {len(self.members_coordinates)}\nNon-members: {len(self.non_members_coordinates)}"
        )
    
    def on_closing(self):
        """Handle window close - save all marked images"""
        print("\n" + "="*50)
        print("Saving all data...")
        
        # Save marked images for both modes
        if self.members_coordinates:
            self.save_marked_image("members")
        if self.non_members_coordinates:
            self.save_marked_image("non_members")
        
        print(f"\nSummary:")
        print(f"  Galaxy Members: {len(self.members_coordinates)} points")
        print(f"  Background Objects: {len(self.non_members_coordinates)} points")
        print(f"  Output folder: {os.path.abspath(self.output_folder)}")
        print("="*50)
        
        self.root.destroy()


def main():
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use file dialog to select image
        root = tk.Tk()
        root.withdraw()  # Hide main window
        image_path = filedialog.askopenfilename(
            title="Select PNG Image",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        root.destroy()
        
        if not image_path:
            print("No image selected. Exiting.")
            sys.exit(0)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    # Create main window
    root = tk.Tk()
    app = ImageClickRecorder(root, image_path)
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Run application
    root.mainloop()


if __name__ == "__main__":
    main()
