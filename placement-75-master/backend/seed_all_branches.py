import pymysql

db_config = {'host': 'localhost', 'user': 'root', 'password': 'Anna4@aa', 'database': 'placement_app'}

# Questions for each canonical branch
branch_questions = {
    "IT": [
        ("What is the difference between TCP and UDP?", "TCP is connection-oriented and guarantees delivery order. UDP is connectionless and faster but does not guarantee delivery."),
        ("Explain what a RESTful API is.", "REST is an architectural style for distributed hypermedia systems. API endpoints represent resources, and standard HTTP verbs (GET, POST, PUT, DELETE) define actions on them."),
        ("What is a database index and why is it used?", "An index is a data structure that speeds up data retrieval by allowing the database engine to find rows without scanning the entire table."),
        ("Define the concept of a 'firewall'.", "A firewall is a security system that monitors and controls incoming and outgoing network traffic based on predefined security rules."),
        ("What is the purpose of version control systems like Git?", "Version control tracks changes to code over time, allows collaboration between multiple developers, and enables reverting to previous states."),
        ("Explain what cloud computing is.", "Cloud computing delivers on-demand computing services (servers, storage, databases) over the internet on a pay-as-you-go basis."),
        ("What is the difference between SQL and NoSQL databases?", "SQL databases are relational and use structured schemas. NoSQL databases are non-relational and handle unstructured or semi-structured data with more flexibility."),
    ],
    "AI&DS": [
        ("What is the difference between supervised and unsupervised learning?", "Supervised learning trains models on labeled data to predict outcomes. Unsupervised learning finds patterns in unlabeled data without predefined outputs."),
        ("Define overfitting in machine learning.", "Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on new unseen data."),
        ("What is a neural network?", "A neural network is a series of algorithms that mimic the operations of a human brain to recognize patterns, consisting of layers of interconnected nodes."),
        ("Explain what a confusion matrix is.", "A confusion matrix is a table used to evaluate a classification model by showing true positives, true negatives, false positives, and false negatives."),
        ("What is the purpose of the train/test split?", "The train/test split divides data into a training set to build the model and a test set to evaluate its performance on unseen data, preventing overfitting."),
        ("What is gradient descent?", "Gradient descent is an optimization algorithm used to minimize the cost function in machine learning by iteratively adjusting parameters in the direction of the steepest descent."),
        ("How does a decision tree work?", "A decision tree splits data into subsets based on feature values, creating a tree-like model of decisions. Each leaf node represents a class label or prediction."),
    ],
    "CSBS": [
        ("What is the role of a business analyst in software development?", "A business analyst bridges the gap between IT and business by understanding requirements, analyzing data, and translating needs into technical specifications."),
        ("Explain blockchain in simple terms.", "Blockchain is a distributed, immutable ledger that records transactions across a peer-to-peer network without a central authority."),
        ("What is agile methodology?", "Agile is an iterative project management approach that delivers software in incremental sprints, focusing on collaboration, customer feedback, and adaptability."),
        ("Define a use case diagram.", "A use case diagram is a UML diagram that represents the functional requirements of a system by showing actions performed by actors."),
        ("What is data warehousing?", "A data warehouse is a central repository that integrates data from multiple sources for reporting, analysis, and business intelligence purposes."),
        ("What is the difference between ERP and CRM?", "ERP manages internal business processes like finance and supply chain. CRM manages relationships and interactions with customers to improve sales and retention."),
        ("Explain the concept of 'Big Data'.", "Big Data refers to extremely large datasets that traditional software cannot process efficiently. Key characteristics are Volume, Velocity, and Variety."),
    ],
    "ECE": [
        ("What is the difference between an analog and digital signal?", "Analog signals are continuous waveforms that vary smoothly over time. Digital signals are discrete binary representations using 0s and 1s."),
        ("Define VLSI design.", "VLSI (Very Large Scale Integration) design involves creating integrated circuits by combining thousands of transistors onto a single chip."),
        ("What is modulation and why is it used?", "Modulation is the process of encoding information onto a carrier signal for transmission. It is used to transmit signals over long distances efficiently."),
        ("Explain the working of a transistor.", "A transistor is a semiconductor device used to amplify or switch electronic signals. It controls current flow between collector and emitter via the base terminal."),
        ("What is the role of an ADC in an embedded system?", "An ADC (Analog-to-Digital Converter) translates real-world analog signals (like temperature) into digital values that a microcontroller can process."),
        ("What is impedance matching?", "Impedance matching ensures maximum power transfer between source and load by making the source and load impedances equal, reducing signal reflections."),
        ("Describe the working principle of a capacitor.", "A capacitor stores electrical energy in an electric field between two conductive plates separated by an insulating dielectric material."),
    ],
    "EEE": [
        ("What is Kirchhoff's Voltage Law?", "KVL states that the algebraic sum of all voltages in a closed loop or mesh is equal to zero."),
        ("Explain the working principle of a transformer.", "A transformer transfers electrical energy between two circuits through electromagnetic induction, changing the voltage level while keeping power approximately constant."),
        ("What is a PLC and where is it used?", "A PLC (Programmable Logic Controller) is an industrial computer used to automate and control machinery and processes in manufacturing environments."),
        ("Define power factor and its significance.", "Power factor is the ratio of real power to apparent power. A low power factor means inefficient power usage; utilities charge penalties for low power factor loads."),
        ("What is the difference between AC and DC motors?", "AC motors run on alternating current and are used for constant speed applications. DC motors use direct current and offer better speed control."),
        ("What is a circuit breaker?", "A circuit breaker is an automatically operated switch designed to protect circuits from damage caused by excess current from an overload or short circuit."),
        ("Explain what a relay is.", "A relay is an electrically operated switch that uses an electromagnet to open or close contacts, allowing a small current to control a larger one."),
    ],
    "AEI": [
        ("What is the principle behind a PID controller?", "A PID controller uses Proportional, Integral, and Derivative terms to continuously calculate an error value and apply corrections to reach a desired setpoint."),
        ("Define a sensor and a transducer.", "A sensor detects physical changes in an environment. A transducer converts one form of energy to another, such as pressure to electrical signals."),
        ("What is SCADA?", "SCADA (Supervisory Control and Data Acquisition) is a system that monitors and controls industrial processes by collecting real-time data from remote sensors."),
        ("Explain what a feedback control system is.", "A feedback control system uses the output of a process to adjust the input, reducing error and maintaining the process at a desired setpoint."),
        ("What is the role of a DAC in instrumentation?", "A DAC (Digital-to-Analog Converter) translates digital values from a controller into analog signals to drive actuators and other analog devices."),
        ("Define calibration in instrumentation.", "Calibration is the process of configuring an instrument to provide a known measurement within a specified accuracy compared to a standard reference."),
        ("What is a telemetry system?", "A telemetry system collects measurements and transmits data wirelessly from remote locations to a monitoring station for recording and analysis."),
    ],
    "MECH": [
        ("What is the difference between stress and strain?", "Stress is the force applied per unit area in a material. Strain is the deformation or elongation of the material relative to its original length due to stress."),
        ("Explain the working principle of a four-stroke engine.", "A four-stroke engine completes a power cycle in four piston strokes: intake, compression, combustion (power), and exhaust."),
        ("What is thermodynamics?", "Thermodynamics is the branch of physics that deals with heat, work, and temperature, and their relation to energy, entropy, and physical properties of matter."),
        ("Define what CNC machining is.", "CNC (Computer Numerical Control) machining uses pre-programmed computer software to control the movement of factory machinery for precise manufacturing."),
        ("What is the purpose of a flywheel?", "A flywheel stores rotational kinetic energy to smooth out fluctuations in power output, providing a more consistent speed to machinery."),
        ("Explain the concept of thermal efficiency.", "Thermal efficiency is the ratio of work output to the heat energy supplied to a system, indicating how well a heat engine converts heat to useful work."),
        ("What is fatigue failure in materials?", "Fatigue failure occurs when a material fractures after repeated cyclic loading at stress levels lower than its ultimate tensile strength."),
    ],
    "CIVIL": [
        ("What is the difference between a beam and a column?", "A beam is a horizontal structural member that carries transverse loads. A column is a vertical structural member that carries axial compressive loads."),
        ("Define the water-to-cement ratio.", "The water-to-cement ratio (w/c) is the ratio of water to cement in a concrete mix. A lower ratio generally results in stronger and more durable concrete."),
        ("What is RCC (Reinforced Cement Concrete)?", "RCC is a composite material where steel reinforcement bars (rebar) are embedded in concrete to resist tensile stresses, combining the compressive strength of concrete and tensile strength of steel."),
        ("Explain what soil liquefaction is.", "Soil liquefaction occurs when water-saturated soil temporarily loses its strength and behaves like a liquid, typically during an earthquake."),
        ("What is the purpose of surveying in civil engineering?", "Surveying measures and maps land features to plan, design, and construct infrastructure accurately, establishing control points and boundaries."),
        ("Define bearing capacity of soil.", "Bearing capacity is the maximum load per unit area that the soil can support without shear failure or excessive settlement."),
        ("What is the difference between a culvert and a bridge?", "A culvert is a short enclosed channel for directing water flow under a road or embankment. A bridge spans a larger distance (like a river) to carry traffic."),
    ],
}

def seed_all_branches():
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cur:
            for branch, qs in branch_questions.items():
                # Check if branch already has questions
                cur.execute("SELECT COUNT(*) FROM interview_questions WHERE branch=%s", (branch,))
                existing = cur.fetchone()[0]
                if existing >= 7:
                    print(f"  SKIP {branch}: already has {existing} questions")
                    continue
                for question, ideal in qs:
                    cur.execute("INSERT INTO interview_questions (branch, question, ideal_answer) VALUES (%s, %s, %s)", (branch, question, ideal))
                print(f"  ADDED {len(qs)} questions for {branch}")
        conn.commit()
        print("Done seeding all branches.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    seed_all_branches()
