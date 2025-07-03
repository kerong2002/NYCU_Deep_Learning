'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''
import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os

def run_experiment(param_definitions, variables):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py_path = os.path.join(script_dir, 'main.py')
    command = [sys.executable, main_py_path]

    params = {}
    for name in param_definitions.keys():
        key_name = f"--{name}"
        var_name = name.replace('-', '_') + '_var'
        if var_name in variables:
            params[key_name] = variables[var_name].get()

    for key, value in params.items():
        if value:
            if key == '--hidden-dims':
                command.append(key)
                command.extend(value.split())
            else:
                command.append(key)
                command.append(value)

    print(f"Executing command: {' '.join(command)}")

    if sys.platform == "win32":
        subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', ' '.join(command)])
    elif sys.platform == "darwin":
        subprocess.Popen(command)
    else:
        try:
            subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f"{' '.join(command)}; exec bash"])
        except FileNotFoundError:
            subprocess.Popen(['xterm', '-e', f"{' '.join(command)}"])

# --- GUI Setup ---
root = tk.Tk()
root.title("DLP Lab1 - Experiment Launcher")
root.geometry("560x620")
root.resizable(False, False)

# Style
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 12))
style.configure("TCombobox", font=("Helvetica", 12))

# Main Frame
mainframe = ttk.Frame(root, padding="20 20 20 20")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Title
title_label = ttk.Label(mainframe, text="DLP Lab1 - Experiment Launcher", font=("Helvetica", 16, "bold"))
title_label.grid(column=0, row=0, columnspan=2, pady=(0, 20))

# Parameters
param_definitions = {
    'dataset': {'type': 'dropdown', 'options': ['linear', 'xor'], 'default': 'linear'},
    'epochs': {'type': 'entry', 'default': '1000'},
    'lr': {'type': 'entry', 'default': '0.01'},
    'hidden-dims': {'type': 'entry', 'default': '10 10'},
    'activation': {'type': 'dropdown', 'options': ['sigmoid', 'relu', 'none'], 'default': 'sigmoid'},
    'optimizer': {'type': 'dropdown', 'options': ['sgd', 'gd', 'adam', 'adagrad'], 'default': 'adam'},
    'loss': {'type': 'dropdown', 'options': ['bce', 'mse', 'cross'], 'default': 'bce'},
    'momentum': {'type': 'entry', 'default': '0.0'},
    'seed': {'type': 'entry', 'default': '1'},
    'log-interval': {'type': 'entry', 'default': '1000'}
}

variables = {}
current_row = 1
for name, config in param_definitions.items():
    label = ttk.Label(mainframe, text=f"--{name}:")
    label.grid(column=0, row=current_row, sticky=tk.W, pady=4, padx=8)

    var = tk.StringVar(value=config['default'])
    var_name = name.replace('-', '_') + '_var'
    variables[var_name] = var

    if config['type'] == 'dropdown':
        widget = ttk.Combobox(mainframe, textvariable=var, values=config['options'], width=25)
        widget.state(['readonly'])
    else:
        widget = ttk.Entry(mainframe, textvariable=var, width=28)

    widget.grid(column=1, row=current_row, sticky=(tk.W, tk.E), pady=4)
    current_row += 1

# Run button
run_button = ttk.Button(mainframe, text="🚀 Run Experiment", command=lambda: run_experiment(param_definitions, variables))
run_button.grid(column=1, row=current_row, sticky=tk.E, pady=20)

# Padding
for child in mainframe.winfo_children():
    child.grid_configure(padx=8, pady=4)

# Launch
root.mainloop()
