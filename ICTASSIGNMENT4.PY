# # Program no 1
# class Inventory:
#     def __init__(self, item_number, quantity, cost):
#         self.item_number = item_number
#         self.quantity = quantity
#         self.cost = cost
#         self.total_cost = 0

#     def set_item_number(self, item_number):
#         self.item_number = item_number

#     def set_quantity(self, quantity):
#         self.quantity = quantity
#         self.update_total_cost()

#     def set_cost(self, cost):
#         self.cost = cost
#         self.update_total_cost()

#     def update_total_cost(self):
#         self.total_cost = self.quantity * self.cost

#     def get_item_number(self):
#         return self.item_number

#     def get_quantity(self):
#         return self.quantity

#     def get_cost(self):
#         return self.cost

#     def get_total_cost(self):
#         return self.total_cost

# # Example usage
# item1 = Inventory(23022, 4, 380)

# # Initial values
# print("Initial Values are:")
# print("Item Number:", item1.get_item_number())
# print("Quantity:", item1.get_quantity())
# print("Cost:", item1.get_cost())
# print("Total Cost:", item1.get_total_cost())

# print("\n")

# # Updating values
# item1.set_quantity(19)
# item1.set_cost(200)

# # Updated values
# print("Updated Values are:")
# print("Item Number:", item1.get_item_number())
# print("Quantity:", item1.get_quantity())
# print("Cost:", item1.get_cost())
# print("Total Cost:", item1.get_total_cost())


# PROGRAM NO 2
# class Room_Dimension:
#     def __init__(room, length, width, cost_per_square_foot):
#         room.length = length
#         room.width = width
#         room.cost_per_square_foot = cost_per_square_foot
#     def get_area(room):
#         return room.length * room.width
#     def calculate_total_cost(room):
#         return room.get_area() * room.cost_per_square_foot
#     def printing_details(room):
#         area = room.get_area()
#         total_cost = room.calculate_total_cost()
#         print(f"Area of the room: {area} square feet")
#         print(f"Total cost for carpeting the room: ${total_cost}")

# length = float(input("Enter the Length of the room in feet: "))
# width = float(input("Enter the width of the room in feet: "))
# cost_per_square_foot = float(input("Enter the cost of carpet per square foot: "))
# room = Room_Dimension(length, width, cost_per_square_foot)
# room.printing_details()

#Program no 3
class Vehicle:
    def __init__(vehicle, modelName, registrationNumber, owner):
        vehicle.modelName = modelName
        vehicle.registrationNumber = registrationNumber
        vehicle.owner = owner
    def set_model_name(vehicle, modelName):
        vehicle.modelName = modelName
    def set_registration_number(vehicle, registrationNumber):
        vehicle.registrationNumber = registrationNumber
    def set_owner(vehicle, owner):
        vehicle.owner = owner
    def get_model_name(vehicle):
        return vehicle.modelName
    def get_registration_number(vehicle):
        return vehicle.registrationNumber
    def get_owner(vehicle):
        return vehicle.owner
    def display(vehicle):
        return f"Vehicle Model Name: {vehicle.modelName}, Registration Number: {vehicle.registrationNumber}, Owner: {vehicle.owner}"
class Car(Vehicle):
    def __init__(vehicle, modelName, registrationNumber, owner, trunk):
        super().__init__(modelName, registrationNumber, owner)
        vehicle.trunk = trunk
    def set_trunk(vehicle, trunk):
        vehicle.trunk = trunk
    def get_trunk(vehicle):
        return vehicle.trunk
    def display(vehicle):
        return super().display() + f", Car Trunk Dimension: {vehicle.trunk}"
class Truck(Vehicle):
    def __init__(vehicle, modelName, registrationNumber, owner, cargo_bed):
        super().__init__(modelName, registrationNumber, owner)
        vehicle.cargo_bed = cargo_bed
    def set_cargo_bed(vehicle, cargo_bed):
        vehicle.cargo_bed = cargo_bed

    def get_cargo_bed(vehicle):
        return vehicle.cargo_bed

    def display(vehicle):
        return super().display() + f", Cargo Bed Dimension: {vehicle.cargo_bed}"


# Creating objects
car = Car("Toyota Corolla", "10023ff", "Zaheer Abbasi", "200X200X500mm")
truck = Truck("Bmw M6 ", "DEF456", "Zaheer Ahmed", "2X3X4 meters")

print("CAR DETAILS:-")
print(car.display())

print("TRUCK DETAILS:-")
print(truck.display())