import pygad
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
import base64
import cv2
from sklearn.cluster import KMeans
from sympy.utilities.iterables import multiset_permutations
import subprocess
import json
import os

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
home = os.path.join(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = home+'/static/scene/'
blender_scene = "static\\scene\\RecentSeamlessBackground_camoublend_withPython.blend"
blender_script = "static\\scene\\blender_bakingScript.py"


def fitness_function(ga_instance, solution, sol_idx):
    global fitnesses_from_user
    # print(sol_idx)
    fitness_per_pop = fitnesses_from_user[sol_idx]
    return fitness_per_pop
    # return 0


num_generations = 1
num_parents_mating = 7
sample_sol_pop = 10
num_genes = 5  # k features

ga_instance = None
k_rgba_colours = None


@app.route('/', methods=['GET'])
def index():

    # print(data)
    # ratio = np.random.randn((population,genes))
    return render_template('test.html', data={'start': True})


@app.route('/', methods=['POST'])
def start():
    if request.method == 'POST':

        background_name = 'u'
        k_colours = request.form['colours']

        if k_colours == '' or background_name == '':
            return render_template('index.html', message='Please enter required fields')

        f = request.files['background']

        f = request.files['background']
        imgByte = f.read()
        nparr = np.fromstring(imgByte, np.uint8)
        jpg_as_text = base64.b64encode(imgByte).decode('utf-8')

        nparr = np.fromstring(imgByte, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_feature = img_rgb.reshape(
            ((img_rgb.shape[0]*img_rgb.shape[1]), img_rgb.shape[2]))
        clf = KMeans(n_clusters=int(k_colours))
        clf.fit(img_rgb_feature)

        hist = centroid_histogram(clf)
        bar, colours, Color_percent = plot_colors(hist, clf.cluster_centers_)

        # _p = [t['percent'] for t in Color_percent]
        # _c = [t['color'] for t in Color_percent]
        labels = []
        sizes = []
        Colors = []
        rgbas = []
        for color_dict in Color_percent:
            Colors.append(tuple(color_dict['color']))
            rgbas.append(tuple(np.array(color_dict['color'] + [255.0])/255.0))
            labels.append(color_dict['label'])
            sizes.append(color_dict['percent']*100)
        # argv = [
        #     str(BASE_DIR)+'\\static\\scene\\background.png', str(rgbas)]  # , '[(0.1,0,0,1);(0,0.2,0,1);(0,0,0.3,1);(0.1,0.2,0,1);(0.5,0,0.5,1)]'
        # subprocess.run(["blender", "-b", 'static/scene/Background_camoublend_withPython.blend',
        #                "-x", "1", "-o", "//rendered", "-a", '--enable-autoexec', ] + argv)
        img_camo = cv2.imread(str(BASE_DIR)+'rendered0001.png')
        retval, buffer = cv2.imencode('.png', img_camo)
        img_camo_as_text = base64.b64encode(buffer).decode('utf-8')

        soldier_camo = cv2.imread(
            str(BASE_DIR)+'rendered0002.png')
        retval, buffer = cv2.imencode('.png', soldier_camo)
        soldier_camo_as_text = base64.b64encode(buffer).decode('utf-8')
        data = {'k': k_colours,
                'path': BASE_DIR,
                'img': jpg_as_text,
                'img_camo': img_camo_as_text,
                'soldier_camo': soldier_camo_as_text,
                'Colors': Colors,
                'labels': labels,
                'sizes': sizes}
        return render_template('_analyse_design.html', data=data)
    return render_template('_analyse_design.html')
    pass


@app.route('/analyse_design', methods=['POST'])
def analyseDesign():
    if request.method == 'POST':

        background_name = 'u'
        k_colours = request.form['colours']

        if k_colours == '' or background_name == '':
            return render_template('index.html', message='Please enter required fields')

        f = request.files['background']

        f = request.files['background']
        imgByte = f.read()
        nparr = np.fromstring(imgByte, np.uint8)
        jpg_as_text = base64.b64encode(imgByte).decode('utf-8')

        nparr = np.fromstring(imgByte, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_feature = img_rgb.reshape(
            ((img_rgb.shape[0]*img_rgb.shape[1]), img_rgb.shape[2]))
        clf = KMeans(n_clusters=int(k_colours))
        clf.fit(img_rgb_feature)

        hist = centroid_histogram(clf)
        bar, colours, Color_percent = plot_colors(hist, clf.cluster_centers_)

        # _p = [t['percent'] for t in Color_percent]
        # _c = [t['color'] for t in Color_percent]
        labels = []
        sizes = []
        Colors = []
        rgbas = []
        for color_dict in Color_percent:
            Colors.append(tuple(color_dict['color']))
            rgbas.append(tuple(np.array(color_dict['color'] + [255.0])/255.0))
            labels.append(color_dict['label'])
            sizes.append(color_dict['percent']*100)
        # argv = [
        #     str(BASE_DIR)+'\\static\\scene\\background.png', str(rgbas)]  # , '[(0.1,0,0,1);(0,0.2,0,1);(0,0,0.3,1);(0.1,0.2,0,1);(0.5,0,0.5,1)]'
        # subprocess.run(["blender", "-b", 'static/scene/Background_camoublend_withPython.blend',
        #                "-x", "1", "-o", "//rendered", "-a", '--enable-autoexec', ] + argv)
        img_camo = cv2.imread(str(BASE_DIR)+'/static/scene/rendered0001.png')
        retval, buffer = cv2.imencode('.png', img_camo)
        img_camo_as_text = base64.b64encode(buffer).decode('utf-8')

        soldier_camo = cv2.imread(
            str(BASE_DIR)+'/static/scene/rendered0002.png')
        retval, buffer = cv2.imencode('.png', soldier_camo)
        soldier_camo_as_text = base64.b64encode(buffer).decode('utf-8')
        data = {'k': k_colours,
                'path': BASE_DIR,
                'img': jpg_as_text,
                'img_camo': img_camo_as_text,
                'soldier_camo': soldier_camo_as_text,
                'Colors': Colors,
                'labels': labels,
                'sizes': sizes}
        return render_template('_analyse_design.html', data=data)
        # return render(request, 'base/home.html', context=context)
    pass


@app.route('/t', methods=['POST'])
def startGA():

    global fitnesses_from_user, k_rgba_colours, blender_scene, blender_script
    global ga_instance  # I will suggest you write to file & reload for the sake of multi user
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
        tmp_fitness = np.random.randn(np.math.factorial(5))
        fitnesses_from_user = tmp_fitness/np.sum(tmp_fitness)
        imgByte = f.read()
        nparr = np.fromstring(imgByte, np.uint8)
        jpg_as_text = base64.b64encode(imgByte).decode('utf-8')

        nparr = np.fromstring(imgByte, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_feature = img_rgb.reshape(
            ((img_rgb.shape[0]*img_rgb.shape[1]), img_rgb.shape[2]))
        clf = KMeans(n_clusters=int(colours))
        clf.fit(img_rgb_feature)

        hist = centroid_histogram(clf)
        bar, colours, Color_percent = plot_colors(hist, clf.cluster_centers_)
        _p = [t['percent'] for t in Color_percent]
        _c = [t['color'] for t in Color_percent]

        # permutation n! as the population
        colour = [col for col in multiset_permutations(_c)]
        # 5!, 5
        # print(colour)
        # convert to rgba
        k_rgba_colours = [[tuple(np.array(c + [255.0])/255.0)
                           for c in colour[i]] for i in range(len(colour))]

        initial_population = [per for per in multiset_permutations(_p)]

        # labels = []
        # sizes = []
        # Colors = []
        # rgbas = []
        # for color_dict in Color_percent:
        #     Colors.append(tuple(color_dict['color']))
        #     rgbas.append(tuple(np.array(color_dict['color'] + [255.0])/255.0))
        #     labels.append(color_dict['label'])
        #     sizes.append(color_dict['percent']*100)

    # print(dir(f))
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           initial_population=initial_population)
    # initial_population

    # sample first 10 genes of the 5!
    genes = ga_instance.population[:sample_sol_pop]

    colours = k_rgba_colours[:sample_sol_pop]
    for i, gene in enumerate(genes):
        var = {str(percent): colours[i][j] for j, percent in enumerate(gene)}
        # var = {str(key): colour
        #    for key, colour in zip(gene, colours[i])}

        argv = [BASE_DIR+'gene_{}.png'.format(i), str(var)]
        # subprocess.run(["blender", "-b", blender_scene,
        #               "--python", blender_script, '--'] + argv)
    data = {
        "percent": genes.tolist(),
        "genes": colour[:sample_sol_pop],  # .tolist()
        "img_text": jpg_as_text,
        'generations': ga_instance.generations_completed
    }
    # templates\start_ga_autofitnessUntil_clicked.html
    return render_template('start_ga_autofitnessUntil_clicked.html', data={'population': data})


@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')


# @app.route('/evolve', methods=['POST'])
@app.route('/evolve_mouseOver', methods=['GET'])
def evolve_mouseOver_4fitness():
    global ga_instance, fitnesses_from_user, blender_scene, blender_script, k_rgba_colours

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
    no_maniplable = len(fitnesses_from_user) - \
        len(new_fitness)  # to avoid out of bound error
    start = 0  # np.random.randint(0, no_maniplable)
    fitnesses_from_user[start:len(new_fitness)] = new_fitness
    ga_instance.run()

    proportion = ga_instance.population
    # print(ga_instance.best_solution())
    # print(ga_instance.population)
    # proportion = np.random.rand(population, no_genes)

    # , '[(0.1,0,0,1);(0,0.2,0,1);(0,0,0.3,1);(0.1,0.2,0,1);(0.5,0,0.5,1)]'
    # argv = [str(BASE_DIR)+'/static/scene/background.png', str(rgbas)]
    var = {'0.2': (0.1, 0, 0, 1), '0.4': (0, 0.2, 0, 1), '0.6': (
        0, 0, 0.3, 1), '0.8': (0.1, 0.2, 0, 1), '1.0': (0.5, 0, 0.5, 1)}

    # img_camo = cv2.imread(str(BASE_DIR)+'/static/scene/rendered0001.png')
#         retval, buffer = cv2.imencode('.png', img_camo)
#         img_camo_as_text = base64.b64encode(buffer).decode('utf-8')

#         soldier_camo = cv2.imread(
#             str(BASE_DIR)+'/static/scene/rendered0002.png')
#         retval, buffer = cv2.imencode('.png', soldier_camo)
#         soldier_camo_as_text = base64.b64encode(buffer).decode('utf-8')

    total = np.sum(proportion, axis=1)

    genes = np.array([(per/tot)
                     for per, tot in zip(proportion, total)])[:sample_sol_pop]
    colours = k_rgba_colours[:sample_sol_pop]
    for i, gene in enumerate(genes):
        var = {str(percent): colours[i][j] for j, percent in enumerate(gene)}

        argv = [BASE_DIR+'gene_{}.png'.format(i), str(var)]
        subprocess.run(["blender", "-b", blender_scene,
                        "--python", blender_script, '--'] + argv)
    data = {
        "percent": genes.tolist()
    }
    # formData
    return jsonify(data)
# Route to start the GA
# templates\start_ga_mouseOver_4fitness.html
# templates\start_ga_autofitnessUntil_clicked.html


@app.route('/evolve_clicked', methods=['GET'])
def evolve_autofitnessUntil_clicked():
    global ga_instance, fitnesses_from_user, blender_scene, blender_script, k_rgba_colours

    formData = request.values
    # fitness = request.form['fitness']  # {}
    new_fitness = np.zeros(len(formData))
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
    no_maniplable = len(fitnesses_from_user) - \
        len(new_fitness)  # to avoid out of bound error
    start = 0  # np.random.randint(0, no_maniplable)
    fitnesses_from_user[start:len(new_fitness)] = new_fitness
    ga_instance.run()

    proportion = ga_instance.population
    # print(ga_instance.best_solution())
    # print(ga_instance.population)
    # proportion = np.random.rand(population, no_genes)

    # , '[(0.1,0,0,1);(0,0.2,0,1);(0,0,0.3,1);(0.1,0.2,0,1);(0.5,0,0.5,1)]'
    # argv = [str(BASE_DIR)+'/static/scene/background.png', str(rgbas)]
    var = {'0.2': (0.1, 0, 0, 1), '0.4': (0, 0.2, 0, 1), '0.6': (
        0, 0, 0.3, 1), '0.8': (0.1, 0.2, 0, 1), '1.0': (0.5, 0, 0.5, 1)}

    # img_camo = cv2.imread(str(BASE_DIR)+'/static/scene/rendered0001.png')
#         retval, buffer = cv2.imencode('.png', img_camo)
#         img_camo_as_text = base64.b64encode(buffer).decode('utf-8')

#         soldier_camo = cv2.imread(
#             str(BASE_DIR)+'/static/scene/rendered0002.png')
#         retval, buffer = cv2.imencode('.png', soldier_camo)
#         soldier_camo_as_text = base64.b64encode(buffer).decode('utf-8')

    total = np.sum(proportion, axis=1)

    genes = np.array([(per/tot)
                     for per, tot in zip(proportion, total)])[:sample_sol_pop]
    colours = k_rgba_colours[:sample_sol_pop]
    tmp = []
    for i, gene in enumerate(genes):
        var = [(str(percent), colours[i][j]) for j, percent in enumerate(gene)]
        tmp.append(var)
    dataPath = BASE_DIR+'result.json'
    with open(dataPath, 'w') as fp:
        json.dump(tmp, fp)
    argv = [BASE_DIR+'gene_', dataPath]
    blender_script = BASE_DIR+'blender_bakingScript_faster.py'
    subprocess.run(["blender", "-b", blender_scene,
                    "--python", blender_script, '--'] + argv)
    data = {
        "percent": genes.tolist(),
        "generations": ga_instance.generations_completed
    }
    # formData
    return jsonify(data)


@app.route('/start_ga', methods=['GET'])
def start_genetic_algorithm():
    # best_solution, best_fitness = run_genetic_algorithm()
    best_solution, best_fitness = 0.2, 1
    response = {
        'best_solution': best_solution,
        'best_fitness': best_fitness
    }
    return jsonify(response)

# ###################### helper


def extract_k_colour(fileString):
    pass


def centroid_histogram(clt):
    # get the no of different clusters and create a histogram
    # group to cluster according to the no of pixels assigned
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram,
    # so that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart
    # representing the relative frequency
    # of per colors
    startX = 0
    bar = np.zeros((50, 300, 3), dtype="uint8")
    colours = []
    Color_percent = []

    # loop over the percentage of each cluster
    # and the color of each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
        colours.append(color.astype("uint8").tolist())
        Color_percent.append({"label": color.astype("uint8").tolist(
        ), "color": color.astype("uint8").tolist(), "percent": percent})

    return bar, colours, Color_percent


# Run the Flask app
if __name__ == '__main__':
    app.run()


# nparr = np.fromstring(imgByte, np.uint8)
# img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# cv2.imwrite(str(BASE_DIR)+'/static/scene/background.png', img)
# print(img.shape)
# jpg_as_text = base64.b64encode(imgByte).decode('utf-8')
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_rgb_feature = img_rgb.reshape(
#     ((img_rgb.shape[0]*img_rgb.shape[1]), img_rgb.shape[2]))
# clf = KMeans(n_clusters=int(k))
# clf.fit(img_rgb_feature)
