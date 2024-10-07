# from .context import sample


def foo():
    print("foo")

def dec(func):
    def wrapper():
        print("decorated")
        func()
        print("done")
    return wrapper

print(foo())
print(dec(foo)())

print("---------------")

@dec
def bar():
    print("bar")

print(bar())
