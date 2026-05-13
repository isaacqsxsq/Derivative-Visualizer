# Derivative Visualizer

A visual implementation of numerical derivatives, tangent lines, derivative curves, and the chain rule — built from scratch in pure NumPy and Matplotlib.
Backpropagation is the chain rule applied through a neural network. Visualizing derivatives first makes it intuitive.

---

## Features

- Plot `f(x) = x²` and `f(x) = sin(x)`
- Numerical derivative implementation: `(f(x+h) - f(x)) / h`
- Tangent lines drawn at any point on a curve
- Derivative curve plotted alongside the original function
- Chain rule implemented manually for `f(g(x))`
- All numerical results verified against known analytical answers

---

## Project Structure

```
derivative_visualizer/
├── main.ipynb      # Full derivative visualization pipeline
└── README.md
```

---

## Requirements

- Python 3.10 or higher
- NumPy
- Matplotlib

Install:
```
pip install numpy matplotlib
```

---

## How to Run

```
jupyter notebook main.ipynb
```

Run cells in order — each stage builds on the previous.

---

## What Each Stage Does

### Stage 1 — Plot Functions
```python
def f1(x): return x**2
def f2(x): return np.sin(x)
```
Side by side plots of both functions over `[-3, 3]`.

### Stage 2 — Numerical Derivative
```python
def derivative(f, x, h=0.00001):
    return (f(x + h) - f(x)) / h
```
Approximates the derivative at any point using the limit definition.
Verified against known analytical answers:
```
f1'(2) = 4.000010  expected: 4.0
f2'(0) = 1.000000  expected: 1.0
```
The small error (~0.00001) is numerical error — inherent to finite h.

### Stage 3 — Tangent Lines
```python
def tangent_line(f, x0, x):
    slope = derivative(f, x0)
    return f(x0) + slope * (x - x0)
```
Draws tangent lines at `x = -2, 0, 2` on `f(x) = x²`:
```
x=-2: slope=-4.0  (descending steeply)
x= 0: slope= 0.0  (flat — this is the minimum)
x= 2: slope= 4.0  (ascending steeply)
```

### Stage 4 — Derivative Curve
Plots `f'(x)` across the entire domain alongside `f(x)`.
For `f(x) = x²`, the derivative `f'(x) = 2x` is a straight line through the origin — visually confirms the analytical result.

### Stage 5 — Chain Rule
```python
def chain_rule(f, g, x):
    return derivative(f, g(x)) * derivative(g, x)
```
Implements `d/dx[f(g(x))] = f'(g(x)) * g'(x)` numerically.

Example: `f(x) = x²`, `g(x) = sin(x)` → `f(g(x)) = sin²(x)`
```
Numerical derivative of f(g(x)) at x=1: 0.909297
Chain rule result:                       0.909297
Match: True
```

---

## The Math

**Numerical derivative:**
```
f'(x) ≈ (f(x+h) - f(x)) / h     h = 0.00001
```

**Tangent line at x₀:**
```
y = f(x₀) + f'(x₀) * (x - x₀)
```

**Chain rule:**
```
d/dx [f(g(x))] = f'(g(x)) * g'(x)
```

---

## Numerical Error

The numerical derivative is an approximation — not exact:

| h value | f1'(2) result | Error |
|---|---|---|
| 0.1 | 4.100000 | 0.1 |
| 0.00001 | 4.000010 | 0.000010 |
| 1e-15 | broken | floating point collapse |

`h = 0.00001` is a good balance between accuracy and floating point stability.

---

## Why This Matters for ML

**Gradient descent** updates weights using derivatives:
```python
weight = weight - learning_rate * derivative
```
The derivative tells which direction increases the loss — move the opposite way.

**Backpropagation** is the chain rule applied backwards through every layer:
```
dL/dw1 = dL/dy * dy/dw3 * dw3/dw2 * dw2/dw1
```
Every multiplication is the chain rule. Every `d/dw` is a derivative.
When you call `loss.backward()` in PyTorch — this is exactly what happens inside.

**Tangent line = gradient direction:**
```
slope > 0  → function increases right → move left to decrease loss
slope < 0  → function decreases right → move right to decrease loss
slope = 0  → minimum found → stop training
```