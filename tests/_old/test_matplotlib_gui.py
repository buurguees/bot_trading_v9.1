#!/usr/bin/env python3
"""
Test simple para verificar que matplotlib y tkinter funcionen correctamente.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def test_matplotlib_gui():
    """Test básico de matplotlib en tkinter"""
    
    print("🧪 Test: Matplotlib + Tkinter GUI")
    print("=" * 40)
    
    try:
        # Crear ventana principal
        root = tk.Tk()
        root.title("Test Matplotlib GUI")
        root.geometry("600x400")
        
        # Crear frame
        frame = ttk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Crear figura matplotlib
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Datos de prueba
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 15, 25, 30]
        
        # Plot
        ax.plot(x, y, marker='o', label='Test Data')
        ax.set_title("Test Matplotlib en Tkinter")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Crear canvas
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        print("✅ Matplotlib + Tkinter funcionando correctamente")
        print("✅ Gráfico visible en la ventana")
        
        # Mostrar ventana
        root.mainloop()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_matplotlib_gui()
