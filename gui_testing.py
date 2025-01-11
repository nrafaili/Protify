import tkinter as tk

def main():
    root = tk.Tk()
    # Load your icon as a PhotoImage
    icon = tk.PhotoImage(file="gleg.png")  
    # Set the window icon
    root.iconphoto(True, icon)

    root.title("Custom Icon Example")
    root.geometry("300x200")
    
    label = tk.Label(root, text="This is my custom icon!")
    label.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()

    def build_model_tab(self):
        """
        Create widgets for BaseModelArguments:
            - model_names: user should be able to choose from standard_benchmark
              and possibly select multiple at once.
        """
        ttk.Label(self.model_tab, text="Model Names:").grid(
            row=0, column=0, padx=10, pady=5, sticky="nw"
        )

        # We'll use a Listbox for multiple selection
        self.model_listbox = tk.Listbox(self.model_tab, selectmode="extended", height=10)
        for model_name in standard_benchmark:
            self.model_listbox.insert(tk.END, model_name)
        self.model_listbox.grid(row=0, column=1, padx=10, pady=5, sticky="nw")

        run_button = ttk.Button(self.model_tab, text="Run Model", command=self.run_model)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))