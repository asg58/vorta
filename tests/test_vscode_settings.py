# Test bestand om VS Code settings te demonstreren
def test_function(x, y):
    """Calculate sum of two numbers with proper formatting."""
    result = x + y
    return result


class TestClass:
    """Example class following PEP 8 standards."""
    
    def __init__(self, name):
        self.name = name
    
    def bad_formatting(self):
        """Method with corrected formatting."""
        print(f"Hello {self.name}")


def poorly_formatted_function(x, y, z):
    """Function with proper formatting and type hints."""
    if x > y:
        return z
    else:
        return None
