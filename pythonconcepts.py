users = {
    "aeinstein": {
        "first": "albert",
        "last": "einstein",
        "location": "princeton",
    },
    "mcurie": {
        "first": "marie",
        "last": "curie",
        "location": "paris",
    },
}

for username, user_info in users.items():
    print(f"\nUsername: {username}")
    full_name = f"{user_info['first']} {user_info['last']}"
    location = user_info["location"]

    print(f"\tFull name: {full_name.title()}")
    print(f"\tLocation: {location.title()}")

# --------------------------------------------------------
favorite_languages = {
    "jen": ["python", "rust"],
    "sarah": ["c"],
    "edward": ["rust", "go"],
    "phil": ["python", "haskell"],
}
for name, languages in favorite_languages.items():
    print(f"\n{name.title()}'s favorite languages are:")
    for language in languages:
        print(f"\t{language.title()}")


# ===========================================================#
responses = {}
# Set a flag to indicate that polling is active.
polling_active = True

while polling_active:
    # Prompt for the person's name and response.
    name = input("\nWhat is your name? ")
    response = input("Which mountain would you like to climb someday? ")

    # Store the response in the dictionary.
    responses[name] = response

    # Find out if anyone else is going to take the poll.
    repeat = input("Would you like to let another person respond? (yes/ no) ")
    if repeat == "no":
        polling_active = False

# Polling is complete. Show the results.
print("\n--- Poll Results ---")
for name, response in responses.items():
    print(f"{name} would like to climb {response}.")


# ===========================================================#
def make_pizza(size: int, *toppings: str) -> str:
    """The * is used when we had multiple items"""
    """Summarize the pizza we are about to make."""
    print(f"\nMaking a {size}-inch pizza with the following toppings:")
    for topping in toppings:
        z: str = f"- {topping}"
    return z


print(make_pizza(16, "pepperoni", "mushrooms"))

# --------------------------------------------------------



class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0

    def get_descriptive_name(self):
        long_name = f"{self.year} {self.make} {self.model}"
        return long_name.title()



class Battery:
    def __init__(self, battery_size,battery_type= "volta strong"):
        self.battery_size = battery_size
        self.battey_type = battery_type

    def describe_battery(self):
        print(f"This car has a {self.battery_size}-kWh battery.")


class ElectricCar(Car):
    """Represent aspects of a car, specific to electric vehicles."""

    def __init__(self, make, model, year,battery_size):
        super().__init__(make, model, year)
        self.battery = Battery(battery_size)


my_leaf = ElectricCar('nissan', 'leaf', 2024,45)
print(my_leaf.get_descriptive_name())
my_leaf.battery.describe_battery()


from pathlib import Path
import json


numbers = [2, 133, 5, 7, 11, 13,17]
path1=Path('username3.json')
contents=path1.read_text()
# words = contents.split()
# lines = contents.splitlines()
path1.write_text
data=json.loads(contents)
print(data)

path = Path('numbers.json')
contents = json.dumps(numbers)
path.write_text(contents)

from pathlib import Path
import json
def get_stored_username(path):
    """Get stored username if available."""
    if path.exists():
        contents = path.read_text()
        username = json.loads(contents)
        return username
    else:
        return None

def get_new_username(path):
    """Prompt for a new username."""
    username = input("What is your name? ")
    contents = json.dumps(username)
    path.write_text(contents)
    return username

def greet_user():
    """Greet the user by name."""
    path = Path('username.json')
    username = get_stored_username(path)
    if username:
        print(f"Welcome back, {username}!")
    else:
        username = get_new_username(path)
        print(f"We'll remember you when you come back, {username}!")

greet_user()


