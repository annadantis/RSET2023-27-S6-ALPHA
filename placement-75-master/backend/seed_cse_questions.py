import pymysql
import os

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Anna4@aa',
    'database': 'placement_app'
}

questions = [
    {
        "question": "Define what an object is in the context of C++.",
        "ideal_answer": "An object represents an instance of a class that encapsulates both data (attributes) and methods (behaviors). It acts as a distinct entity that can interact with other parts of the program.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "In object-oriented terms, what is a 'message'?",
        "ideal_answer": "A message is a request sent from one object to another to perform a specific action or calculation. It is conceptually similar to calling a function on an object.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What is the primary role of a class in OOP?",
        "ideal_answer": "A class serves as a blueprint or template for creating objects. It defines the structure, behavior, and properties that all instances of that class will share.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Explain the concept of an 'instance'.",
        "ideal_answer": "An instance is a specific occurrence of an object created from a class. If a class is the blueprint, an instance is the actual house built from that blueprint.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What is a super-class and how does it relate to its children?",
        "ideal_answer": "A super-class (or base class) provides the foundation for derived classes. Sub-classes inherit characteristics from the super-class, allowing for code reuse and hierarchical organization.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Describe how inheritance works in C++.",
        "ideal_answer": "Inheritance allows a child class to acquire properties and methods from a parent class. It enables the child to extend or modify the parent's behavior while maintaining a relationship.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What does 'message protocol' mean for an object?",
        "ideal_answer": "A message protocol refers to the set of valid messages or interface that an object is capable of receiving and responding to.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Explain polymorphism using a simple example.",
        "ideal_answer": "Polymorphism allows different objects to respond to the same message in their own unique way. For example, a '+' operator can add integers or concatenate strings, showing different behaviors for the same operation.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What are instance variables and where are they kept?",
        "ideal_answer": "Instance variables are attributes specific to each object instance, representing its private data. They are defined within the class structure.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What is the difference between an instance variable and a class variable?",
        "ideal_answer": "Instance variables belong to a specific object. Class variables (static members in C++) are shared by all instances of the class and represent shared state.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Define a 'method' in object-oriented programming.",
        "ideal_answer": "A method is a function defined within a class that describes the actions or behaviors an object can perform in response to a message.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What are the roles of constructors and destructors in C++?",
        "ideal_answer": "Constructors initialize an object when it is created, while destructors perform cleanup tasks (like releasing memory) when an object is destroyed.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "How does C++ improve upon the standard C language?",
        "ideal_answer": "C++ is a superset of C that adds object-oriented features, stricter type checking, function overloading, references, and the Standard Template Library (STL).",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What is operator overloading and when is it useful?",
        "ideal_answer": "Operator overloading allows you to redefine how standard operators (like +, -, *) work with user-defined types (classes), making code more intuitive for custom types like Complex Numbers.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Define 'cin' and 'cout' in the context of C++ streams.",
        "ideal_answer": "They are standard stream objects; 'cin' is used for receiving input (standard input) and 'cout' is used for displaying output (standard output).",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Contrast procedural programming with object-oriented programming.",
        "ideal_answer": "Procedural programming focuses on a sequence of steps and global data, like a recipe. OOP focuses on objects containing both data and logic, interacting like characters in a play.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "How can you call a C function from a C++ program?",
        "ideal_answer": "You use the 'extern \"C\"' block around the C function declaration. This prevents C++ name mangling, allowing the linker to find the original C function name.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What is the purpose of the scope resolution operator (::)?",
        "ideal_answer": "The '::' operator allows you to access a global variable if a local one has the same name, or to define methods outside of their class definition.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What are the technical differences between a 'struct' and a 'class' in C++?",
        "ideal_answer": "The only differences are default access levels: 'struct' members and inheritance default to 'public', whereas 'class' defaults to 'private'.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "In how many ways can you initialize an integer with a constant in C++?",
        "ideal_answer": "Primarily two: C-style (int x = 5;) and functional/constructor style (int x(5);).",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "How does C++ exception handling differ from setjmp/longjmp in C?",
        "ideal_answer": "C++ exceptions (try/catch/throw) ensure that destructors for automatic objects are called during stack unwinding, whereas setjmp/longjmp does not perform this cleanup.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Is the code 'delete this;' ever acceptable? What are the risks?",
        "ideal_answer": "It is rarely acceptable. It's risky because if the object wasn't allocated on the heap, it crashes. Also, pointers to that object elsewhere in the program become invalid and dangerous.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What is a default constructor?",
        "ideal_answer": "A constructor that either has no parameters or has default values for all its parameters, allowing an object to be created without explicit arguments.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Explain what a conversion constructor is.",
        "ideal_answer": "It is a constructor that takes exactly one argument, allowing the compiler to implicitly convert an object of that argument's type to an object of the class.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Differentiate between a copy constructor and an assignment operator.",
        "ideal_answer": "A copy constructor creates a new object from an existing one. An assignment operator copies data from one already-existing object to another.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "When is multiple inheritance actually justified in design?",
        "ideal_answer": "It should be used rarely, only when a class logically fits into multiple categories that cannot be modeled by simple inheritance or interfaces alone.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Why is a virtual destructor important in class hierarchies?",
        "ideal_answer": "A virtual destructor ensures that the destructor of a derived class is called when deleting an object through a base class pointer, preventing memory leaks.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Describe the ISA and HASA relationships in class design.",
        "ideal_answer": "ISA refers to inheritance (a Manager is-a Employee). HASA refers to composition or aggregation (a Car has-a Engine).",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "When would you choose a template over inheritance?",
        "ideal_answer": "Templates are better for generic containers (like lists or stacks) where the logic is the same regardless of the data type stored inside.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What are the standard access specifiers in C++ and their defaults?",
        "ideal_answer": "C++ has public, private, and protected. For classes, the default is private; for structs, the default is public.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What is data encapsulation and why is it beneficial?",
        "ideal_answer": "Encapsulation hides the internal state and requires all interaction to be performed through an object's methods, protecting data integrity and simplifying usage.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Briefly explain the Unified Modeling Language (UML).",
        "ideal_answer": "UML is a standardized visual language used to document and design the architecture of complex software systems, particularly object-oriented ones.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "Contrast a 'shallow copy' with a 'deep copy'.",
        "ideal_answer": "A shallow copy only copies pointers, leading both objects to share the same data. A deep copy duplicates the actual data, giving both objects their own independent copies.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What does the 'static' keyword mean for a class member?",
        "ideal_answer": "A static member is shared by all instances of a class. There is only one copy of that member regardless of how many objects are created.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "How is memory managed in C versus C++?",
        "ideal_answer": "In C, you use malloc() and free(). In C++, you use the 'new' and 'delete' operators, which also handle constructor and destructor calls.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    },
    {
        "question": "What is the result of using a 'const' keyword with an object?",
        "ideal_answer": "The 'const' keyword ensures that the object's value cannot be modified after it is initialized, providing a form of read-only protection.",
        "branch": "COMPUTER SCIENCE ENGINEERING"
    }
]

def seed_database():
    try:
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            # First, double check and delete any lingering questions for this branch
            cursor.execute("DELETE FROM interview_questions WHERE branch IN ('COMPUTER SCIENCE ENGINEERING', 'CSE', 'Common');")
            
            sql = "INSERT INTO interview_questions (branch, question, ideal_answer) VALUES (%s, %s, %s)"
            for q in questions:
                cursor.execute(sql, (q['branch'], q['question'], q['ideal_answer']))
        
        connection.commit()
        print(f"Successfully seeded {len(questions)} questions.")
    except Exception as e:
        print(f"Error seeding database: {e}")
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    seed_database()
