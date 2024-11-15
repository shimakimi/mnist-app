from sklearn.datasets import fetch_openml

x, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False)
print("Fetch OpenML bcompleted")

print(f"len(x) = {len(x)}, len(y) = {len(y)}")

print("X[0]")
print(x[0])

print("y[0]")
print(y[0])

X0 = x[0]
print(f"len(X0) = {len(X0)}")

print("X0")
for i, v in enumerate(X0):
    print(f"%3d " % v, end="")
    if i % 28 == 27:
        print()