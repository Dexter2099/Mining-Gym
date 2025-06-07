import tkinter as tk
from tkinter import filedialog, messagebox
import threading

import mGym_GymRun as gymrun
import mGym_DefSchdRun as defrun


def run_in_thread(target, *args):
    th = threading.Thread(target=target, args=args, daemon=True)
    th.start()


def start_train():
    try:
        episodes = int(train_episodes_var.get())
    except ValueError:
        messagebox.showerror("Input Error", "Episodes must be an integer")
        return
    config = train_config_var.get() or "config_extend.txt"
    render = train_render_var.get()
    run_in_thread(gymrun.main, "train", episodes, None, config, render)


def start_play():
    model = play_model_var.get()
    if not model:
        messagebox.showerror("Input Error", "Model path required")
        return
    try:
        episodes = int(play_episodes_var.get())
    except ValueError:
        messagebox.showerror("Input Error", "Episodes must be an integer")
        return
    config = play_config_var.get() or "config_extend.txt"
    render = play_render_var.get()
    run_in_thread(gymrun.main, "play", episodes, model, config, render)


def start_scheduler():
    try:
        episodes = int(sched_episodes_var.get())
        algo = int(sched_algo_var.get())
    except ValueError:
        messagebox.showerror("Input Error", "Episodes and algorithm choice must be integers")
        return
    config = sched_config_var.get() or "config_extend.txt"
    run_in_thread(defrun.main, episodes, algo, config)


root = tk.Tk()
root.title("Mining Gym GUI")

# Training frame
train_frame = tk.LabelFrame(root, text="Train RL Model", padx=5, pady=5)
train_frame.pack(fill="x", padx=10, pady=5)

train_episodes_var = tk.StringVar(value="2")
train_config_var = tk.StringVar(value="config_extend.txt")
train_render_var = tk.StringVar(value="console")

tk.Label(train_frame, text="Episodes:").grid(row=0, column=0, sticky="e")
train_episodes_entry = tk.Entry(train_frame, textvariable=train_episodes_var, width=10)
train_episodes_entry.grid(row=0, column=1)

tk.Label(train_frame, text="Config file:").grid(row=1, column=0, sticky="e")
train_config_entry = tk.Entry(train_frame, textvariable=train_config_var, width=25)
train_config_entry.grid(row=1, column=1)

tk.Label(train_frame, text="Render mode:").grid(row=2, column=0, sticky="e")
train_render_menu = tk.OptionMenu(train_frame, train_render_var, "console", "human")
train_render_menu.grid(row=2, column=1, sticky="w")

train_button = tk.Button(train_frame, text="Start Training", command=start_train)
train_button.grid(row=3, columnspan=2, pady=5)

# Play frame
play_frame = tk.LabelFrame(root, text="Play RL Model", padx=5, pady=5)
play_frame.pack(fill="x", padx=10, pady=5)

play_model_var = tk.StringVar()
play_episodes_var = tk.StringVar(value="1")
play_config_var = tk.StringVar(value="config_extend.txt")
play_render_var = tk.StringVar(value="console")

tk.Label(play_frame, text="Model path:").grid(row=0, column=0, sticky="e")
play_model_entry = tk.Entry(play_frame, textvariable=play_model_var, width=25)
play_model_entry.grid(row=0, column=1)

model_browse = tk.Button(play_frame, text="Browse", command=lambda: play_model_var.set(filedialog.askopenfilename()))
model_browse.grid(row=0, column=2)

tk.Label(play_frame, text="Episodes:").grid(row=1, column=0, sticky="e")
play_episodes_entry = tk.Entry(play_frame, textvariable=play_episodes_var, width=10)
play_episodes_entry.grid(row=1, column=1)

tk.Label(play_frame, text="Config file:").grid(row=2, column=0, sticky="e")
play_config_entry = tk.Entry(play_frame, textvariable=play_config_var, width=25)
play_config_entry.grid(row=2, column=1)

tk.Label(play_frame, text="Render mode:").grid(row=3, column=0, sticky="e")
play_render_menu = tk.OptionMenu(play_frame, play_render_var, "console", "human")
play_render_menu.grid(row=3, column=1, sticky="w")

play_button = tk.Button(play_frame, text="Start Playing", command=start_play)
play_button.grid(row=4, columnspan=3, pady=5)

# Scheduler frame
sched_frame = tk.LabelFrame(root, text="Run Classical Scheduler", padx=5, pady=5)
sched_frame.pack(fill="x", padx=10, pady=5)

sched_episodes_var = tk.StringVar(value="10")
sched_algo_var = tk.StringVar(value="1")
sched_config_var = tk.StringVar(value="config_extend.txt")

tk.Label(sched_frame, text="Episodes:").grid(row=0, column=0, sticky="e")
sched_episodes_entry = tk.Entry(sched_frame, textvariable=sched_episodes_var, width=10)
sched_episodes_entry.grid(row=0, column=1)

tk.Label(sched_frame, text="Algorithm #:").grid(row=1, column=0, sticky="e")
sched_algo_entry = tk.Entry(sched_frame, textvariable=sched_algo_var, width=10)
sched_algo_entry.grid(row=1, column=1)

tk.Label(sched_frame, text="Config file:").grid(row=2, column=0, sticky="e")
sched_config_entry = tk.Entry(sched_frame, textvariable=sched_config_var, width=25)
sched_config_entry.grid(row=2, column=1)

sched_button = tk.Button(sched_frame, text="Run Scheduler", command=start_scheduler)
sched_button.grid(row=3, columnspan=2, pady=5)

root.mainloop()
