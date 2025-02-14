import sympy

x = sympy.Symbol('x')
y = sympy.Symbol('y')
f_x = (x + 1) / ((x + 1) ** 2 + y ** 2) - (x - 1) / ((x - 1) ** 2 + y ** 2)
f_y = y / ((x + 1) ** 2 + y ** 2) - y / ((x - 1) ** 2 + y ** 2)

norm = sympy.simplify(f_x**2 + f_y**2)

print('Norm is', norm)

print(norm.subs({x: -1, y: 0.1}))
print(norm.subs({x: 1, y: -0.1}))
print(norm.subs({x: 0, y: 2}))
print(norm.subs({x: 0, y: -12}))
