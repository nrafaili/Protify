import tkinter as tk
from tkinter import ttk


class SettingsGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Game Settings")
        self.master.geometry("500x350")

        # Dictionary to store all settings
        self.settings = {}

        # Create a Notebook widget that holds multiple tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True)

        # Create frames for each settings tab
        self.general_tab = ttk.Frame(self.notebook)
        self.graphics_tab = ttk.Frame(self.notebook)
        self.audio_tab = ttk.Frame(self.notebook)
        
        # Add tabs to the notebook
        self.notebook.add(self.general_tab, text="General")
        self.notebook.add(self.graphics_tab, text="Graphics")
        self.notebook.add(self.audio_tab, text="Audio")

        # Build each tab
        self.build_general_tab()
        self.build_graphics_tab()
        self.build_audio_tab()

        # Add a "Save All" button below the notebook
        apply_button = ttk.Button(master, text="Save All", command=self.save_settings)
        apply_button.pack(side="bottom", pady=10)

    def build_general_tab(self):
        """Initialize all widgets for the General settings tab."""
        # Example: Username
        ttk.Label(self.general_tab, text="Username:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings["username"] = tk.StringVar(value="Player1")
        username_entry = ttk.Entry(self.general_tab, textvariable=self.settings["username"])
        username_entry.grid(row=0, column=1, padx=10, pady=5)

        # Example: Difficulty (dropdown)
        ttk.Label(self.general_tab, text="Difficulty:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings["difficulty"] = tk.StringVar(value="Normal")
        difficulty_combo = ttk.Combobox(
            self.general_tab, 
            textvariable=self.settings["difficulty"], 
            values=["Easy", "Normal", "Hard", "Nightmare"]
        )
        difficulty_combo.grid(row=1, column=1, padx=10, pady=5)

        # Example: Enable tutorials (checkbox)
        ttk.Label(self.general_tab, text="Enable Tutorials:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings["tutorials_enabled"] = tk.BooleanVar(value=True)
        tutorial_check = ttk.Checkbutton(self.general_tab, variable=self.settings["tutorials_enabled"])
        tutorial_check.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Add a "Run General" button at the bottom of the tab
        run_button = ttk.Button(self.general_tab, text="Run General", command=self.run_general)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def build_graphics_tab(self):
        """Initialize all widgets for the Graphics settings tab."""
        # Example: Screen Resolution
        ttk.Label(self.graphics_tab, text="Resolution:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings["resolution"] = tk.StringVar(value="1920x1080")
        resolution_combo = ttk.Combobox(
            self.graphics_tab, 
            textvariable=self.settings["resolution"], 
            values=["1920x1080", "1280x720", "1600x900", "2560x1440"]
        )
        resolution_combo.grid(row=0, column=1, padx=10, pady=5)

        # Example: Fullscreen
        ttk.Label(self.graphics_tab, text="Fullscreen:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings["fullscreen"] = tk.BooleanVar(value=True)
        fullscreen_check = ttk.Checkbutton(self.graphics_tab, variable=self.settings["fullscreen"])
        fullscreen_check.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Example: Graphics Quality (dropdown)
        ttk.Label(self.graphics_tab, text="Graphics Quality:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings["graphics_quality"] = tk.StringVar(value="High")
        quality_combo = ttk.Combobox(
            self.graphics_tab,
            textvariable=self.settings["graphics_quality"],
            values=["Low", "Medium", "High", "Ultra"]
        )
        quality_combo.grid(row=2, column=1, padx=10, pady=5)

        # Example: Frame Rate Limit (Spinbox)
        ttk.Label(self.graphics_tab, text="Frame Limit:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings["frame_limit"] = tk.IntVar(value=60)
        fr_spinbox = ttk.Spinbox(
            self.graphics_tab, 
            from_=30, 
            to=240, 
            textvariable=self.settings["frame_limit"]
        )
        fr_spinbox.grid(row=3, column=1, padx=10, pady=5)

        # Add a "Run Graphics" button at the bottom of the tab
        run_button = ttk.Button(self.graphics_tab, text="Run Graphics", command=self.run_graphics)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def build_audio_tab(self):
        """Initialize all widgets for the Audio settings tab."""
        # Example: Master Volume
        ttk.Label(self.audio_tab, text="Master Volume:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings["master_volume"] = tk.DoubleVar(value=100)
        volume_scale = ttk.Scale(
            self.audio_tab, 
            from_=0, 
            to=100, 
            orient="horizontal", 
            variable=self.settings["master_volume"]
        )
        volume_scale.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        # Example: Mute
        ttk.Label(self.audio_tab, text="Mute:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings["mute"] = tk.BooleanVar(value=False)
        mute_check = ttk.Checkbutton(self.audio_tab, variable=self.settings["mute"])
        mute_check.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Example: Music Volume
        ttk.Label(self.audio_tab, text="Music Volume:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings["music_volume"] = tk.DoubleVar(value=80)
        music_scale = ttk.Scale(
            self.audio_tab, 
            from_=0, 
            to=100, 
            orient="horizontal", 
            variable=self.settings["music_volume"]
        )
        music_scale.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Add a "Run Audio" button at the bottom of the tab
        run_button = ttk.Button(self.audio_tab, text="Run Audio", command=self.run_audio)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def run_general(self):
        """
        Called when the user clicks "Run General".
        Here, you can do whatever you need to do with the general settings.
        """
        print("=== Running General Code ===")
        print(f"Username: {self.settings['username'].get()}")
        print(f"Difficulty: {self.settings['difficulty'].get()}")
        print(f"Tutorials Enabled: {self.settings['tutorials_enabled'].get()}")
        print("General code executed!")
        print("============================\n")

    def run_graphics(self):
        """
        Called when the user clicks "Run Graphics".
        """
        print("=== Running Graphics Code ===")
        print(f"Resolution: {self.settings['resolution'].get()}")
        print(f"Fullscreen: {self.settings['fullscreen'].get()}")
        print(f"Graphics Quality: {self.settings['graphics_quality'].get()}")
        print(f"Frame Limit: {self.settings['frame_limit'].get()}")
        print("Graphics code executed!")
        print("============================\n")

    def run_audio(self):
        """
        Called when the user clicks "Run Audio".
        """
        print("=== Running Audio Code ===")
        print(f"Master Volume: {self.settings['master_volume'].get()}")
        print(f"Mute: {self.settings['mute'].get()}")
        print(f"Music Volume: {self.settings['music_volume'].get()}")
        print("Audio code executed!")
        print("==========================\n")

    def save_settings(self):
        """Retrieve current values from all widget variables and do something with them."""
        # For demonstration, just print them to the console.
        print("=== Saving All Settings ===")
        for key, var in self.settings.items():
            print(f"{key}: {var.get()}")
        print("===========================\n")

def main():
    root = tk.Tk()
    app = SettingsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
