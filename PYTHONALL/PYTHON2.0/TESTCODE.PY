import unittest
import my_code  # Replace with the name of your module

class MyCodeTest(unittest.TestCase):

    def test_calculate_area_of_rectangle(self):
        length = 5
        width = 3
        expected_area = 15

        result = my_code.calculate_area_of_rectangle(length, width)
        self.assertEqual(expected_area, result)

    def test_calculate_area_of_triangle(self):
        base = 6
        height = 4
        expected_area = 12

        result = my_code.calculate_area_of_triangle(base, height)
        self.assertEqual(expected_area, result)

    def test_calculate_average_of_three_numbers(self):
        numbers = [2, 4, 6]
        expected_average = 4

        result = my_code.calculate_average_of_three_numbers(numbers)
        self.assertEqual(expected_average, result)

if __name__ == '__main__':
    unittest.main()
