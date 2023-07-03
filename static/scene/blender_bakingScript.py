import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]
bakeFileName = argv[-2]
colours = eval(argv[-1])  # [1:-1])
print(colours)
obj = bpy.context.active_object
# You can choose your texture size (This will be the de bake image)
image_name = obj.name + '_BakedTexture'
img = bpy.data.images.new(image_name, 1024, 1024)

# pass dictionary

# {'0.2':(0.1, 0, 0, 1), '0.4':(0, 0.2, 0, 1), '0.6':(0, 0, 0.3, 1), '0.8':(0.1, 0.2, 0, 1), '1.0':(0.5, 0, 0.5, 1)}


print('Here in script')
# Due to the presence of any multiple materials, it seems necessary to iterate on all the materials, and assign them a node + the image to bake.

stepSize = 1/len(colours)
startPosition = stepSize
# assign the default colour
col_keys = list(colours.keys())

for mat in obj.data.materials:

    colour_ramp = mat.node_tree.nodes["ColorRamp"].color_ramp

    colour_ramp.elements[0].color = colours[col_keys[0]]
    colour_ramp.elements[0].position = float(col_keys[0])
    for i in range(1, len(col_keys)):
        # create new colour & assign
        colour = colours[col_keys[i]]
        # startPosition += stepSize
        colour_ramp.elements.new(position=float(col_keys[i]))
        colour_ramp.elements[i].color = colour  # (0,1,0,1)
    mat.use_nodes = True  # Here it is assumed that the materials have been created with nodes, otherwise it would not be possible to assign a node for the Bake, so this step is a bit useless
    nodes = mat.node_tree.nodes
    texture_node = nodes.new('ShaderNodeTexImage')
    texture_node.name = 'Bake_node'
    texture_node.select = True
    nodes.active = texture_node
    texture_node.image = img  # Assign the image to the node

bpy.context.view_layer.objects.active = obj
bpy.ops.object.bake(type='DIFFUSE', save_mode='EXTERNAL')

img.save_render(filepath=argv[0])  # 'baked_scripted.png'

# In the last step, we are going to delete the nodes we created earlier
for mat in obj.data.materials:
    for n in mat.node_tree.nodes:
        if n.name == 'Bake_node':
            mat.node_tree.nodes.remove(n)
