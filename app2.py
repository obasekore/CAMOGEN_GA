import pygad
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Define the fitness function


def evaluate_fitness(ga_instance, solution, sol_idx):
    # Add your fitness evaluation logic here
    fitness_value = ...
    return fitness_value

# Define the main function to run the Genetic Algorithm


# # Run the GA
# ga_instance.run()

# # Retrieve the best solution
# best_solution = ga_instance.best_solution()
# best_fitness = ga_instance.best_solution()[1]

# # Return the results
# return best_solution, best_fitness

# Flask backend
ENV = 'dev'
app = Flask(__name__)

if ENV == 'dev':
    app.debug = True
else:
    app.debug = False

fitnesses_from_user = []


def fitness_function(ga_instance, solution, sol_idx):
    global fitnesses_from_user
    # print(sol_idx)
    fitness_per_pop = fitnesses_from_user[sol_idx]
    return fitness_per_pop
    # return 0


num_generations = 1
num_parents_mating = 7
sol_per_pop = 10
num_genes = 5  # k features

ga_instance = None


@app.route('/', methods=['GET'])
def index():
    global ga_instance
    population = 10  # best 10
    no_genes = 5  # or features or k
    color_channel = 3

    proportion = np.random.rand(population, no_genes)

    colour = np.random.randint(0, 255, size=(
        population, no_genes, color_channel))

    total = np.sum(proportion, axis=1)

    genes = np.array([(per/tot) for per, tot in zip(proportion, total)])
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           initial_population=genes)
    data = {
        "percent": genes.tolist(),
        "genes": colour  # .tolist()
    }
    print(data)
    # ratio = np.random.randn((population,genes))
    return render_template('index.html', data={'population': data})


@app.route('/', methods=['POST'])
def startGA():
    if request.method == 'POST':
        # f = request.files['background']
        # background_name = request.form['background']
        background_name = 'u'
        colours = request.form['colours']
        # print(customer, dealer, rating, comments)
        if colours == '' or background_name == '':
            return render_template('index.html', message='Please enter required fields')

            # return render_template('success.html')
        f = request.files['background']
        f.stream.seek(0)
        print(f.filename)
        content = ""
        for line in f.stream.readlines():
            # print(bytearray(line))  # .decode("UTF-8"))
            content += str(line)  # .decode("UTF-8")

        # f.save(secure_filename(f.filename))
    # initial_population
    global ga_instance
    # print(dir(f))
    # ga_instance = pygad.GA(num_generations=num_generations,
    #                        num_parents_mating=num_parents_mating,
    #                        fitness_func=fitness_function,
    #                        initial_population=sol_per_pop)
    # UPLOAD IMAGE
    #
    img = Image.frombytes(content)
    return img.data()
    pass


@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')


# @app.route('/evolve', methods=['POST'])
@app.route('/evolve', methods=['GET'])
def evolve():
    global ga_instance, fitnesses_from_user

    formData = request.values
    # fitness = request.form['fitness']  # {}
    new_fitness = np.zeros(len(formData)-1)
    for key in formData:
        value = formData.get(key)
        if key == "submit":
            # print(key)
            continue
        # print(key, value)
        idx = int(key.split(']')[0].split('[')[-1])
        new_fitness[idx] = float(value)  # NB: the fitnesses are not ordered
    # if request.method == 'POST':
    #     fitness = request.form['fitness']

    fitnesses_from_user = new_fitness
    ga_instance.run()

    population = 10  # len(value)-1
    no_genes = 5

    proportion = ga_instance.population
    # print(ga_instance.best_solution())
    # print(ga_instance.population)
    # proportion = np.random.rand(population, no_genes)
    total = np.sum(proportion, axis=1)

    genes = np.array([(per/tot) for per, tot in zip(proportion, total)])
    data = {
        "percent": genes.tolist()
    }
    # formData
    return jsonify(data)
# Route to start the GA


@app.route('/start_ga', methods=['GET'])
def start_genetic_algorithm():
    # best_solution, best_fitness = run_genetic_algorithm()
    best_solution, best_fitness = 0.2, 1
    response = {
        'best_solution': best_solution,
        'best_fitness': best_fitness
    }
    return jsonify(response)


# Run the Flask app
if __name__ == '__main__':
    app.run()
