import sympy

x = sympy.Symbol('x')
y = sympy.Symbol('y')
f_x = (x + 1) / pow(((x + 1) ** 2 + y ** 2), 3/2) - (x - 1) / pow(((x - 1) ** 2 + y ** 2), 3/2)
f_y = y / pow(((x + 1) ** 2 + y ** 2), 3/2) - y / pow(((x - 1) ** 2 + y ** 2), 3/2)

norm = sympy.simplify(f_x**2 + f_y**2)

print('Norm is', norm)

print(norm.subs({x: -1, y: 0.1}))
print(norm.subs({x: 1, y: -0.1}))
print(norm.subs({x: 0, y: 2}))
print(norm.subs({x: 0, y: -12}))
