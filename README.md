![Uploading Github_Banner.png…]()


# Image analysis for nanoscale particles

This project aims to develop methods for the segmentation, classification ands 3D-augmentation of SEM images from nanoscale materials.

---

# 📜 Coding Rules

To ensure consistency, readability, and maintainability across the project, please adhere to the following coding rules:

## 1. **Docstrings**
   - **Every file** must start with a **docstring** that briefly describes the purpose of the file.
   - **Every class, function, and method** must also start with a **docstring** that explains its purpose and describes all arguments.

   Example:
   ```python
   """
   This module contains functions for data visualization.
   """
   ```

## 2. **Imports and Global Variables**
   - **All imports** must be defined at the **top of the file**.
   - **Global variables** (if necessary) should also be declared at the **top of the file**.

   Example:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   GLOBAL_CONSTANT = 42
   ```

## 3. **Avoid Code Repetition**
   - **Do not repeat code**. Instead, encapsulate recurring tasks in **functions** or **methods**.
   - If a piece of code is used more than once, it should be refactored into a reusable function.

   Example:
   ```python
   def calculate_average(data):
       return sum(data) / len(data)
   ```

## 4. **File Organization**
   - **Classes, functions, and modules** should be grouped by purpose in **individual files**.
   - For example, all visualization-related functions should be placed in a file named `plotting.py`.

   Example:
   ```
   project/
   ├── main.py
   ├── plotting.py
   ├── data_processing.py
   ```

## 5. **Descriptive Variable Names**
   - Use **long, descriptive names** for variables to improve readability.
   - Avoid abbreviations unless they are widely understood.
   - Variable names should be **completely lowercase** with underscores for separation.

   Example:
   ```python
   # Good
   outlet_air_relative_humidity = 0.5

   # Bad
   RH_out = 0.5
   ```

## 6. **Inline Comments**
   - Use **inline comments** sparingly and only when necessary to explain complex or non-obvious code.
   - The code should be self-explanatory through good naming and structure.

   Example:
   ```python
   # Calculate the average temperature
   average_temp = sum(temperatures) / len(temperatures)
   ```

## 7. **Commit Messages**
   - Write **short but precise commit messages** that clearly describe the changes made.
   - Avoid vague messages like "fixed bug" or "updated code".

   Example:
   ```python
   git commit -m "Add function to calculate average temperature"
   ```
**By following these rules, we can maintain a clean, consistent, and professional codebase that is easy to understand and collaborate on. Happy coding!** 🚀

---
