class Reventh:
    def __init__(self, name, surname, age, location):
        self.name = name
        self.surname = surname
        self.age = age
        self.location = location

    def introduction(self):
        print("My name is " + self.name + ' ' + self.surname + ". I am " + self.age + " years old and i live in " + self.location)

R1 = Reventh("Reventh", "Thiruvallur", "27", "Malmö")
R1.introduction()