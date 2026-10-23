import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import joblib
import pandas as pd
import math

# Load trained model
model = joblib.load(
    Path(__file__).with_name("housing_model.pkl")
)

# Colors
BG = "#F1F5F9"
WHITE = "#FFFFFF"
TEXT = "#172033"
BLUE = "#2563EB"
GREEN = "#15803D"
GRAY = "#64748B"

# Window
window = tk.Tk()
window.title("California Housing Price Predictor")
window.geometry("720x680")
window.minsize(680, 640)
window.configure(bg=BG)

# Main container
main = tk.Frame(window, bg=BG, padx=30, pady=20)
main.pack(fill="both", expand=True)

# Heading
tk.Label(
    main,
    text="California Housing Predictor",
    font=("Arial", 24, "bold"),
    bg=BG,
    fg=TEXT
).pack(pady=(5, 8))

tk.Label(
    main,
    text="Enter housing details to estimate a house price",
    font=("Arial", 13),
    bg=BG,
    fg=GRAY
).pack(pady=(0, 20))

# White input card
card = tk.Frame(
    main,
    bg=WHITE,
    padx=25,
    pady=25,
    highlightbackground="#CBD5E1",
    highlightthickness=1
)
card.pack(fill="x")

features = [
    "Median Income",
    "House Age",
    "Average Rooms",
    "Average Bedrooms",
    "Population",
    "Average Occupancy",
    "Latitude",
    "Longitude"
]

columns = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]

defaults = [
    "5", "25", "5", "1",
    "1200", "3", "34", "-118"
]

entries = {}

# Two-column input layout
for i, (feature, default) in enumerate(
    zip(features, defaults)
):
    row = i // 2
    col = i % 2

    field = tk.Frame(card, bg=WHITE)
    field.grid(
        row=row,
        column=col,
        padx=12,
        pady=12,
        sticky="ew"
    )

    tk.Label(
        field,
        text=feature,
        font=("Arial", 13, "bold"),
        bg=WHITE,
        fg=TEXT
    ).pack(anchor="w", pady=(0, 6))

    entry = tk.Entry(
        field,
        font=("Arial", 15),
        bg="#F8FAFC",
        fg=TEXT,
        insertbackground=TEXT,
        relief="solid",
        bd=1,
        width=19
    )
    entry.pack(fill="x", ipady=8)
    entry.insert(0, default)

    entries[feature] = entry

card.columnconfigure(0, weight=1)
card.columnconfigure(1, weight=1)


# Predict price
def predict_price():
    try:
        values = [
            float(entries[f].get())
            for f in features
        ]

        if not all(math.isfinite(v) for v in values):
            raise ValueError

        data = pd.DataFrame(
            [values],
            columns=columns
        )

        prediction = model.predict(data)[0]
        price = prediction * 100000

        result_label.config(
            text=f"${price:,.2f}",
            fg=GREEN
        )

        status_label.config(
            text="Estimated House Price"
        )

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numbers in all fields."
        )

    except Exception as e:
        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


# Clear fields
def clear_fields():
    for entry in entries.values():
        entry.delete(0, tk.END)

    result_label.config(
        text="--",
        fg=TEXT
    )

    status_label.config(
        text="Enter values to get a prediction"
    )


# Buttons
button_frame = tk.Frame(main, bg=BG)
button_frame.pack(pady=22)

# Use labels as buttons for readable Mac colors
predict_button = tk.Label(
    button_frame,
    text="PREDICT HOUSE PRICE",
    font=("Arial", 14, "bold"),
    bg=BLUE,
    fg=WHITE,
    padx=25,
    pady=14,
    cursor="hand2"
)
predict_button.pack(side="left", padx=8)
predict_button.bind(
    "<Button-1>",
    lambda event: predict_price()
)

clear_button = tk.Label(
    button_frame,
    text="CLEAR FIELDS",
    font=("Arial", 14, "bold"),
    bg="#CBD5E1",
    fg=TEXT,
    padx=25,
    pady=14,
    cursor="hand2"
)
clear_button.pack(side="left", padx=8)
clear_button.bind(
    "<Button-1>",
    lambda event: clear_fields()
)

# Results area
result_card = tk.Frame(
    main,
    bg=WHITE,
    padx=20,
    pady=20
)
result_card.pack(fill="x")

status_label = tk.Label(
    result_card,
    text="Enter values to get a prediction",
    font=("Arial", 13),
    bg=WHITE,
    fg=GRAY
)
status_label.pack()

result_label = tk.Label(
    result_card,
    text="--",
    font=("Arial", 28, "bold"),
    bg=WHITE,
    fg=TEXT
)
result_label.pack(pady=8)


# Model performance window
def show_performance():
    try:
        path = Path(__file__).with_name("cv_results.csv")
        results = pd.read_csv(path)

        performance = tk.Toplevel(window)
        performance.title("Model Performance")
        performance.geometry("950x750")
        performance.configure(bg="#F1F5F9")

        tk.Label(
            performance,
            text="Random Forest Model Performance",
            font=("Arial", 21, "bold"),
            bg="#F1F5F9",
            fg="#172033"
        ).pack(pady=15)

        averages = results[["MAE", "MSE", "R2"]].mean()

        summary = (
            f"Average MAE: {averages['MAE']:.4f}     "
            f"Average MSE: {averages['MSE']:.4f}     "
            f"Average R²: {averages['R2']:.4f}"
        )

        tk.Label(
            performance,
            text=summary,
            font=("Arial", 13, "bold"),
            bg="#F1F5F9",
            fg="#172033"
        ).pack(pady=10)

        fig, axes = plt.subplots(
            3, 1, figsize=(9, 8)
        )

        metrics = [
            ("MAE", "Mean Absolute Error", "#3B82F6"),
            ("MSE", "Mean Squared Error", "#F59E0B"),
            ("R2", "R² Score", "#16A34A")
        ]

        for ax, (column, title, color) in zip(
            axes, metrics
        ):
            ax.bar(
                results["Fold"].astype(str),
                results[column],
                color=color
            )

            ax.set_title(title)
            ax.set_xlabel("Fold")
            ax.set_ylabel(column)
            ax.grid(axis="y", alpha=0.2)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig, master=performance
        )
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        performance.protocol(
            "WM_DELETE_WINDOW",
            lambda: (plt.close(fig), performance.destroy())
        )

    except Exception as e:
        messagebox.showerror(
            "Performance Error",
            str(e)
        )


# Model Performance button
performance_button = tk.Label(
    main,
    text="VIEW MODEL PERFORMANCE",
    font=("Arial", 13, "bold"),
    bg="#334155",
    fg="white",
    padx=25,
    pady=12,
    cursor="hand2"
)

performance_button.pack(pady=10)

performance_button.bind(
    "<Button-1>",
    lambda event: show_performance()
)


window.mainloop()
