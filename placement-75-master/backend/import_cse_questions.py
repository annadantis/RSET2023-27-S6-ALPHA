from database import SessionLocal
from sqlalchemy import text
import json

questions = [
    {
        "question": "Define an object in C++ or Object-Oriented Programming (OOP).",
        "ideal_answer": "An object is a self-contained unit that encapsulates both data (attributes) and behavior (methods). It serves as a concrete instance of a class, representing a specific entity with its own state and functional capabilities."
    },
    {
        "question": "Explain the concept of 'message passing' between objects.",
        "ideal_answer": "Message passing is the process by which objects interact. One object sends a request (a message) to another object to perform a specific task or return information. In most programming languages, this is implemented as a method call or function invocation on an object."
    },
    {
        "question": "What is a class in OOP and what role does it play?",
        "ideal_answer": "A class is a blueprint or template for creating objects. It defines the structure (data members) and behavior (member functions) that all objects created from that class will possess. It acts as a logical grouping for related functionality."
    },
    {
        "question": "Define an 'instance' of a class.",
        "ideal_answer": "An instance is a specific, individual object realized from a class blueprint. While a class defines the structure, the instance contains actual data values stored in memory."
    },
    {
        "question": "In inheritance, what is a super-class or base class?",
        "ideal_answer": "A super-class (or base class) is the parent class from which other classes (sub-classes) inherit properties and methods. it provides a foundation of common characteristics that can be reused and extended by more specialized classes."
    },
    {
        "question": "Explain inheritance and how derived classes can modify inherited traits.",
        "ideal_answer": "Inheritance allows a child class to acquire the properties and methods of a parent class. Derived classes can extend the parent's functionality or override specific behaviors (methods) to suit their specialized needs while maintaining a fundamental relationship with the base class."
    },
    {
        "question": "What does the 'message protocol' of an object represent?",
        "ideal_answer": "The message protocol of an object is the complete set of messages (or method signatures) to which that object can respond. This defines the object's external interface and how other parts of the system can interact with it."
    },
    {
        "question": "Define polymorphism and provide a conceptual example.",
        "ideal_answer": "Polymorphism is the ability of different objects to respond to the same interface in a way that is appropriate to their specific type. For example, a '+' operator might perform integer addition for numbers but string concatenation for text, hiding the underlying complexity from the user."
    },
    {
        "question": "What are instance variables and where are they stored?",
        "ideal_answer": "Instance variables represent the unique data or state of an individual object. They are defined within a class but belong to the specific memory space allocated for each instance of that class."
    },
    {
        "question": "How do class variables (static) differ from instance variables?",
        "ideal_answer": "Class variables (declared as static) are shared by all instances of a class. There is only one copy of a class variable regardless of how many objects are created, whereas each instance has its own unique copy of instance variables."
    },
    {
        "question": "Explain what a 'method' is in the context of a class.",
        "ideal_answer": "A method is a function defined within a class that defines a behavior or action the class can perform. Methods typically operate on the instance variables of the object they belong to."
    },
    {
        "question": "Describe the roles of constructors and destructors in C++.",
        "ideal_answer": "Constructors are special member functions called automatically when an object is created, used to initialize members and allocate resources. Destructors are called when an object is destroyed to perform cleanup, such as releasing memory or closing file handles."
    },
    {
        "question": "What are the primary differences between C and C++ when used as procedural languages?",
        "ideal_answer": "In procedural mode, C++ acts as an enhanced version of C. It introduces stricter type checking, allows variable declarations anywhere in the code, supports references in addition to pointers, and provides inline functions as a safer alternative to macros."
    },
    {
        "question": "Define operator overloading and explain its utility.",
        "ideal_answer": "Operator overloading allows developers to redefine how standard C++ operators (like +, -, *, or ==) behave when used with custom class objects. This makes user-defined types more intuitive to use and improves code readability."
    },
    {
        "question": "Explain the purpose of cin and cout in C++.",
        "ideal_answer": "cin and cout are predefined objects in the iostream library used for standard input and output. cin is typically associated with the keyboard, while cout is associated with the console screen."
    },
    {
        "question": "Contrast the procedural and object-oriented programming paradigms.",
        "ideal_answer": "Procedural programming focuses on a sequence of steps or functions that manipulate data separately. Object-oriented programming bundles data and the logic that operates on it into objects, focusing on the interaction between these entities."
    },
    {
        "question": "How can you link C++ code to C functions?",
        "ideal_answer": "You can link C++ to C by using the 'extern \"C\"' linkage specification. This tells the C++ compiler to disable name mangling for those functions, allowing the linker to find the symbols in C object files correctly."
    },
    {
        "question": "Describe the use of the scope resolution operator (::).",
        "ideal_answer": "The scope resolution operator (::) is used to access global variables hidden by local ones, to define class methods outside the class declaration, and to specify which namespace or class a particular identifier belongs to."
    },
    {
        "question": "What are the differences between a C++ struct and a class regarding access specifiers?",
        "ideal_answer": "In C++, the only difference is the default access level. In a struct, members and inheritance default to 'public'. In a class, members and inheritance default to 'private'. Otherwise, they possess the same capabilities (methods, inheritance, etc.)."
    },
    {
        "question": "What are the common ways to initialize an integer with a constant in C++?",
        "ideal_answer": "An integer can be initialized using C-style assignment (int x = 10;), constructor-style notation (int x(10);), or uniform initialization using braces (int x{10};) which was introduced in C++11."
    },
    {
        "question": "How does C++ exception handling differ from setjmp/longjmp regarding object destruction?",
        "ideal_answer": "C++ exception handling (try-catch-throw) is 'exception safe' because it performs stack unwinding, which automatically calls destructors for local objects as the stack is cleared. setjmp/longjmp perform a raw jump that bypasses these destructors, potentially leading to resource leaks."
    },
    {
        "question": "What are the potential risks with using 'delete this'?",
        "ideal_answer": "Using 'delete this' is dangerous if the object was not allocated on the heap (using new). It also makes any subsequent access to the object's pointers or members by the calling program invalid, which frequently leads to memory corruption or crashes."
    },
    {
        "question": "What is a default constructor?",
        "ideal_answer": "A default constructor is a constructor that can be called without any arguments. It either has no parameters at all or all its parameters have default values."
    },
    {
        "question": "Define a conversion constructor.",
        "ideal_answer": "A conversion constructor is a constructor that can be called with a single argument of a different type. The compiler uses it to implicitly convert an object of the argument type into an object of the class type."
    },
    {
        "question": "Explain the difference between a copy constructor and an overloaded assignment operator.",
        "ideal_answer": "A copy constructor creates a brand new object as a copy of an existing one. An overloaded assignment operator (=) copies the state of one existing object into another existing object that has already been initialized."
    },
    {
        "question": "When is multiple inheritance a suitable design choice?",
        "ideal_answer": "Multiple inheritance is used when a class naturally 'is-a' specialization of more than one distinct base type. While it can be complex, it is useful for modeling objects that have multifaceted roles that cannot be captured by simple inheritance alone."
    },
    {
        "question": "Why should base class destructors be declared as virtual?",
        "ideal_answer": "Declaring a base class destructor as virtual ensures that when an object is deleted through a base class pointer, the derived class destructor is called correctly. Without it, only the base class destructor might run, causing memory leaks in the derived part of the object."
    },
    {
        "question": "Explain the 'ISA' vs 'HASA' relationships in object-oriented design.",
        "ideal_answer": "ISA represents inheritance (a Car is a Vehicle). HASA represents composition or aggregation (a Car has an Engine). ISA is implemented by deriving one class from another, while HASA is implemented by including an instance of one class as a member of another."
    },
    {
        "question": "When is using a template preferred over a base class for generic components?",
        "ideal_answer": "Templates are preferred when you need a generic container (like a List or Stack) that can hold any data type, regardless of its position in an inheritance hierarchy. Base classes are better when polymorphism and common behavior across types are required."
    },
    {
        "question": "Explain data encapsulation and its importance in software development.",
        "ideal_answer": "Data encapsulation bundles data and methods together while restricting direct access to the internal state. This hides implementation details, reduces complexity, and allows the internal logic to change without breaking external code that relies on the class interface."
    },
    {
        "question": "What is the process of deriving classes, and what relationship does it establish?",
        "ideal_answer": "Deriving a class involves creating a sub-class from a super-class (base class). This establishes an 'is-a' relationship, where the sub-class inherits the features of the base class while potentially adding specialized members of its own."
    },
    {
        "question": "What are the pros and cons of multiple inheritance?",
        "ideal_answer": "Pros include the ability to model complex real-world relationships and combine behaviors from multiple parents. Cons include increased complexity, the 'diamond problem' (ambiguity when two parents share a common ancestor), and harder maintenance."
    },
    {
        "question": "Explain polymorphism in an inheritance hierarchy.",
        "ideal_answer": "Polymorphism allows a collection of objects of different derived types to be treated as if they were objects of a single base type. The correct overridden method is automatically invoked at runtime based on the actual object type."
    },
    {
        "question": "Distinguish between the static and const keywords in C++.",
        "ideal_answer": "Static indicates that a member belongs to the class itself rather than any specific instance. Const indicates that a variable's value or an object's state cannot be modified after initialization."
    },
    {
        "question": "How does memory management (allocation/deallocation) differ between C and C++?",
        "ideal_answer": "C uses malloc() and free() functions, which only handle raw bytes. C++ uses 'new' and 'delete' operators, which not only allocate memory but also automatically call headers/constructors and destructors respectively for objects."
    },
    {
        "question": "What is the Unified Modeling Language (UML)?",
        "ideal_answer": "UML is a standardized visual language used to specify, visualize, and document the design of software systems, particularly object-oriented architectures through diagrams like class diagrams and sequence diagrams."
    },
    {
        "question": "Explain the difference between a shallow copy and a deep copy.",
        "ideal_answer": "A shallow copy copies only the member values, meaning pointers in both objects will point to the same memory location. A deep copy allocates new memory and duplicates the actual data pointed to, ensuring the two objects are completely independent."
    }
]

def import_questions():
    db = SessionLocal()
    try:
        print(f"Importing {len(questions)} CSE interview questions...")
        for q in questions:
            db.execute(text(
                "INSERT INTO interview_questions (branch, question, ideal_answer) VALUES ('CSE', :q, :a)"
            ), {"q": q["question"], "a": q["ideal_answer"]})
        db.commit()
        print("Successfully imported CSE questions.")
    except Exception as e:
        print(f"Error during import: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    import_questions()
