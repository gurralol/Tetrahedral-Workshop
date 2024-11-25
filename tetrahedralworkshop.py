bl_info = {
    "name": "Tetrahedral Workshop",
    "blender": (2, 80, 0),
    "category": "Physics",
}

import bpy
import bmesh
import mathutils
import math
from mathutils.bvhtree import BVHTree
import re

#numSubsteps = 6
#gravity = mathutils.Vector((0.0, 0.0, -0.03)) / numSubsteps
objects = []
currentPosition = []
previousPosition = []
originalPosition = []
invMass = []
edgeLengths = []
bend_constraint_triples = []
restVol = []
restVolDots = []
grads = [(mathutils.Vector((0.0, 0.0, 0.0))), (mathutils.Vector((0.0, 0.0, 0.0))), (mathutils.Vector((0.0, 0.0, 0.0))), (mathutils.Vector((0.0, 0.0, 0.0)))]
volIdOrder = [(1,3,2),(0,2,3),(0,3,1),(0,1,2)]

edgeCompliance = 100.0
volumeCompliance = 0.0

def dot_product(vec1, vec2):
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c

def vectLengthSquared(vector):
    a0 = vector[0]
    a1 = vector[1]
    a2 = vector[2]
    return a0 * a0 + a1 * a1 + a2 * a2

def get_tet_volume(nr):
    id0 = restVol[nr][0]
    id1 = restVol[nr][1]
    id2 = restVol[nr][2]
    id3 = restVol[nr][3]
    
    v0 = objects[0].data.vertices[id1].co.copy() - objects[0].data.vertices[id0].co.copy()
    v1 = objects[0].data.vertices[id2].co.copy() - objects[0].data.vertices[id0].co.copy()
    v2 = objects[0].data.vertices[id3].co.copy() - objects[0].data.vertices[id0].co.copy()
    vC = cross(v0, v1)
    
    return dot_product(vC, v2) / 6.0

def save_position():
    originalPosition.clear()
    for i in range (len(objects)):
        for j in range (len(objects[i].data.vertices)):
            originalPosition.append(objects[i].data.vertices[j].co.copy())
    return

def reset_position():
    for i in range (len(objects)):
        for j in range (len(objects[i].data.vertices)):
            objects[i].data.vertices[j].co = originalPosition[j]
    return

def num_to_range(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

def init_physics():
    invMass.clear()
    restVol.clear()
    restVolDots.clear()
    edgeLengths.clear()
    bend_constraint_triples.clear()
    previousPosition.clear()
    currentPosition.clear()
    
    # Store inverted mass
    for i in range (len(objects)):
        for j in range (len(objects[i].data.vertices)):
            invMass.append(0.0)
    
    # Store tet IDs
    for i in range (len(objects)):
        for face in range (len(objects[i].data.polygons)):
            if len(objects[i].data.polygons[face].vertices) == 4:
                id0, id1, id2, id3 = objects[i].data.polygons[face].vertices
                restVol.append((id0, id1, id2, id3))
    
    # Store rest volume
    for i in range (len(objects)):
        for j in range (len(restVol)):
            restVolDots.append(get_tet_volume(j))
    
    # Store rest distance
    for i in range (len(objects)):
        for j in range (len(objects[i].data.edges)):
            id0, id1 = objects[i].data.edges[j].vertices
            pos0 = objects[i].data.vertices[id0].co.copy()
            pos1 = objects[i].data.vertices[id1].co.copy()
            vector = pos0 - pos1
            distance = vector.length

            edgeLength = distance
            
            edgeLengths.append((id0, id1, distance))
            
    # Store rest bend
    for obj in range (len(objects)):
        for face in objects[obj].data.polygons:
            if len(face.vertices) == 3:  # Ensure the face is a triangle
                i, j, k = face.vertices  # Indices of the vertices in the triangle
                
                # Get vertex positions
                pos_i = objects[obj].data.vertices[i].co.copy()
                pos_j = objects[obj].data.vertices[j].co.copy()
                pos_k = objects[obj].data.vertices[k].co.copy()
                
                # Calculate the angle at vertex j between vectors j->i and j->k
                vec_ij = (pos_i - pos_j).normalized()
                vec_jk = (pos_k - pos_j).normalized()
                rest_angle = vec_ij.angle(vec_jk)
                
                # Add to bend constraints
                bend_constraint_triples.append((i, j, k, rest_angle))
    
    return

def get_animated_bmesh(obj):
    if obj.type != 'MESH':
        raise ValueError("Object must be of type 'MESH'")
    
    # Create a BMesh from the object's current mesh data
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
    
    # Apply the object's world matrix to the BMesh
    for vert in bm.verts:
        vert.co = obj.matrix_world @ vert.co
    
    return bm

def solve_pre(sdt):
    gravity = bpy.data.scenes[0].gravity.copy()
    gravity /= 25

    gravity /= bpy.data.scenes[0].render.fps
    gravity /= props.substeps
    gravity /= props.collisionIterations
    
    # Create bmesh from meshes with collision enabled
    bmCollisions = bmesh.new()
    for i in bpy.data.objects:
        if i.type == 'MESH':
            for j in i.modifiers:
                if j.type == 'COLLISION':
                    bmCollisions = get_animated_bmesh(i)
    bvhTreeCollisions = BVHTree.FromBMesh(bmCollisions)
    
    # Solve
    for i in range (len(objects)):
        iterations = range(props.collisionIterations)
        for its in iterations:
            currentPosition.clear()
            for j in range (len(objects[i].data.vertices)):
                currentPosition.append(objects[i].data.vertices[j].co.copy())
                
                if len(previousPosition) == 0:
                    continue
                
                # Velocity and gravity
                velocity = currentPosition[j] - previousPosition[j]
                
                displacement = velocity + gravity
                """
                displacement /= bpy.data.scenes[0].render.fps
                displacement /= props.substeps
                displacement /= props.collisionIterations
                """
                
                if displacement.length > 10.0 or displacement.length < 0.0001:
                    continue
                
                displacement *= 0.98
                objects[i].data.vertices[j].co += displacement

                world_pos = objects[i].matrix_world @ objects[i].data.vertices[j].co.copy()
                location, normal, index, distance = bvhTreeCollisions.find_nearest(world_pos)
                
                if distance is not None and distance < props.collisionRadius:
                    # Apply a normal push to resolve penetration
                    push_strength = 1.0
                    push_direction = normal.normalized() * (props.collisionRadius - distance) * push_strength
                    objects[i].data.vertices[j].co = objects[i].matrix_world.inverted() @ (world_pos + push_direction)
                    
                    # Apply friction to tangential velocity component
                    velocity_at_vertex = velocity
                    normal_component = velocity_at_vertex.project(normal)  # Velocity perpendicular to plane
                    tangential_component = velocity_at_vertex - normal_component  # Velocity parallel to plane

                    # Reduce tangential velocity by the friction coefficient
                    tangential_component *= (1 - num_to_range(props.friction, 0, 1, 1, 0))
                    velocity = normal_component + tangential_component  # Update velocity with friction
                    objects[i].data.vertices[j].co -= velocity
                    
                
                
                # Pin vertices to animation
                try: 
                    if objects[i].vertex_groups[props.pinGroup].weight(j) > 0.9:
                        objects[i].data.vertices[j].co = originalPosition[j]
                except:
                    pass
            
            try:
                previousPosition[:] = currentPosition
            except:
                pass
            bmCollisions.free()
            
    return

def solve(sdt):
    solve_edges(sdt)
#    solve_bend(sdt)
    solve_volumes(sdt)
    return

def solve_post():
    previousPosition[:] = currentPosition
    currentPosition.clear()
    return

def solve_edges(sdt):
    alpha = props.distanceStiffness / sdt / sdt
    
    for obj in range (len(objects)):
        
        iterations = range(props.distanceIterations)
        for its in iterations:
            for i, j, distance in edgeLengths:
                pos_i = objects[obj].data.vertices[i].co.copy()
                pos_j = objects[obj].data.vertices[j].co.copy()
                w0 = 0.5
                w1 = 0.5
                w = w0 + w1
                if w == 0.0:
                    continue
                
                vector = (pos_i - pos_j)
                if vector.length < 0.0001 or vector.length > 10.0:
                    continue
                vectorLength = vectLengthSquared(vector)
                current_distance = math.sqrt(vectorLength)
    #            current_distance = (pos_i - pos_j).length
                
                if current_distance == 0:
                    continue
                
                vector *= 1.0 / current_distance
                
                direction = (pos_i - pos_j) / current_distance

                C = current_distance - distance
                
                s = -C / (w * alpha)
                
                displacement = vector * ((s * w0) * props.distanceWeight)
                displacement *= props.distanceDamping
                
#                objects[obj].data.vertices[i].co += vector * ((s * w0) * props.distanceWeight)
#                objects[obj].data.vertices[j].co += vector * ((-s * w1) * props.distanceWeight)

                if displacement.length > 10.0 or displacement.length < 0.0001:
                    continue
                
                objects[obj].data.vertices[i].co += displacement
                objects[obj].data.vertices[j].co -= displacement
            
    return

def solve_bend(sdt):
    for obj in range (len(objects)):
        for i, j, k, target_angle in bend_constraint_triples:
            pos_i = objects[obj].data.vertices[i].co
            pos_j = objects[obj].data.vertices[j].co
            pos_k = objects[obj].data.vertices[k].co

            vec_ij = (pos_i - pos_j).normalized()
            vec_jk = (pos_k - pos_j).normalized()

            if vec_ij.length == 0.0 or vec_jk.length == 0.0:
                continue

            current_angle = vec_ij.angle(vec_jk)
            
            # Compute angle difference
            angle_difference = current_angle - target_angle
            if abs(angle_difference) > 0.001:
                correction_strength = 1
                correction = angle_difference * correction_strength

                # Compute rotation axis
                rotation_axis = vec_ij.cross(vec_jk).normalized()
                if rotation_axis.length == 0.0:
                    continue  # Skip if axis is degenerate

                # Compute rotation matrix
                rotation = mathutils.Matrix.Rotation(-correction, 4, rotation_axis)

                # Rotate the vertices relative to the shared vertex j
                pos_i_corrected = rotation @ (pos_i - pos_j) + pos_j
                pos_k_corrected = rotation @ (pos_k - pos_j) + pos_j

                objects[obj].data.vertices[i].co = pos_i_corrected
                objects[obj].data.vertices[k].co = pos_k_corrected
    return

def solve_volumes(sdt):
    alpha = props.volumeStiffness / sdt / sdt
    
    for obj in range (len(objects)):
        
        iterations = range(props.volumeIterations)
        for its in iterations:
            for i in range (len(restVol)):
                
                w = 0.0
                
                for j in range(4):
                    id0 = restVol[i][volIdOrder[j][0]]
                    id1 = restVol[i][volIdOrder[j][1]]
                    id2 = restVol[i][volIdOrder[j][2]]
                    
                    temp0 = objects[obj].data.vertices[id1].co.copy() - objects[obj].data.vertices[id0].co.copy()
                    temp1 = objects[obj].data.vertices[id2].co.copy() - objects[obj].data.vertices[id0].co.copy()
                    crossList = cross(temp0, temp1)
                    crossV = mathutils.Vector((crossList[0], crossList[1], crossList[2]))
                    if crossV.length == 0.0:
                        continue
                    crossV /= 6.0
                    grads[j] = crossV
                    
                    w += -1.0 * vectLengthSquared(crossV)
                
                vol = get_tet_volume(i)
                restVolx = restVolDots[i]
                C = vol - restVolx
                
                s = -C / (w * alpha)
                
                for j in range(4):
                    id = restVol[i][j]
                    displacement = grads[j] * (s * (w / 4))
                    displacement *= props.volumeDamping
#                    objects[obj].data.vertices[id].co += grads[j] * (s * (w / 4))
                    
                    if displacement.length > 10000.0 or displacement.length < 0.0001:
                        continue
                    
                    objects[obj].data.vertices[id].co += displacement
            
    return

def simulate(scene):
    dt = 1 / bpy.data.scenes[0].render.fps
    
    sdt = dt / props.substeps
    
    sRange = range(props.substeps)
    
    for steps in sRange:
        for i in objects:
            solve_pre(sdt)
            
        for i in objects:
            solve(sdt)
            
        for i in objects:
            solve_post()
    
    return

def on_playback_start(scene):
    init_physics()
    bpy.app.handlers.frame_change_pre.append(simulate)
    print("Playback started.")
    return
    
def on_playback_stop(scene):
    bpy.app.handlers.frame_change_pre.clear()
    print("Playback stopped.")
    return

def init_render(scene):
#    init_physics()
    print("Initialized render.")
    return

def on_render(scene):
    simulate(0)
    print("Rendered frame.")
    return

def on_click(scene):
    
    # Only show panel on objects in the physicsObjectsList list.
    try:
        if bpy.context.object.type == 'MESH':
            if bpy.context.object in objects and not TetrahedralWorkshopPanel.is_registered:
                bpy.utils.register_class(TetrahedralWorkshopPanel)
            elif bpy.context.object not in objects and TetrahedralWorkshopPanel.is_registered:
                bpy.utils.unregister_class(TetrahedralWorkshopPanel)
        elif TetrahedralWorkshopPanel.is_registered:
            bpy.utils.unregister_class(TetrahedralWorkshopPanel)
        
        # Only show "Tetrahedral Workshop" button on meshes.
        if bpy.context.object.type == 'MESH' and not TetrahedralWorkshop.is_registered:
            bpy.utils.register_class(TetrahedralWorkshop)
        elif not bpy.context.object.type == 'MESH' and TetrahedralWorkshop.is_registered:
            bpy.utils.unregister_class(TetrahedralWorkshop)
    except:
        pass
        
    return

class ResetPositionButton(bpy.types.Operator):
    bl_idname = "object.reset_position_button"
    bl_label = "Reset Position"
    
    def execute(self, context):       
        reset_position()
        return {'FINISHED'}
    
class InputProperties(bpy.types.PropertyGroup):
    substeps: bpy.props.IntProperty(
        name="",
        description="A floating-point value",
        default=6,
        min=0,   # Minimum value
        max=100, # Maximum value
        step=1,    # Step size for increments
    )
    distanceIterations: bpy.props.IntProperty(
        name="",
        description="A floating-point value",
        default=2,
        min=0,   # Minimum value
        max=100, # Maximum value
        step=1,    # Step size for increments
    )
    volumeIterations: bpy.props.IntProperty(
        name="",
        description="A floating-point value",
        default=2,
        min=0,   # Minimum value
        max=100, # Maximum value
        step=1,    # Step size for increments
    )
    collisionIterations: bpy.props.IntProperty(
        name="",
        description="A floating-point value",
        default=1,
        min=0,   # Minimum value
        max=100, # Maximum value
        step=1,    # Step size for increments
    )
    
    distanceWeight: bpy.props.FloatProperty(
        name="",
        description="A floating-point value",
        default=1.0,
        min=0.0,   # Minimum value
        max=100.0, # Maximum value
        step=10,    # Step size for increments
        precision=2  # Number of decimal places
        )
    distanceStiffness: bpy.props.FloatProperty(
        name="",
        description="A floating-point value",
        default=0.0001,
        min=0.0,   # Minimum value
        max=100.0, # Maximum value
        step=10,    # Step size for increments
        precision=64  # Number of decimal places
        )
    distanceDamping: bpy.props.FloatProperty(
        name="",
        description="A floating-point value",
        default=1.0,
        min=0.0,   # Minimum value
        max=100.0, # Maximum value
        step=10,    # Step size for increments
        precision=2  # Number of decimal places
        )
        
    volumeStiffness: bpy.props.FloatProperty(
        name="",
        description="A floating-point value",
        default=1e-11,
        min=0.0,   # Minimum value
        max=100.0, # Maximum value
        step=10,    # Step size for increments
        precision=64  # Number of decimal places
        )
    volumeDamping: bpy.props.FloatProperty(
        name="",
        description="A floating-point value",
        default=1.0,
        min=0.0,   # Minimum value
        max=100.0, # Maximum value
        step=10,    # Step size for increments
        precision=2  # Number of decimal places
        )
        
    collisionRadius: bpy.props.FloatProperty(
        name="",
        description="A floating-point value",
        default=0.01,
        min=0.0,   # Minimum value
        max=10.0, # Maximum value
        step=1,    # Step size for increments
        precision=4  # Number of decimal places
        )
    friction: bpy.props.FloatProperty(
        name="",
        description="A floating-point value",
        default=1.0,
        min=0.0,   # Minimum value
        max=1.0, # Maximum value
        step=10,    # Step size for increments
        precision=2  # Number of decimal places
        )
        
    pinGroup: bpy.props.StringProperty(
        name="",
        description="",
        default=""
        )

class TetrahedralWorkshopPanel(bpy.types.Panel):
    bl_label = "Tetrahedral Workshop"
    bl_idname = "OBJECT_PT_TETRAHEDRALWORKSHOPPANEL"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "physics"

    def draw(self, context):
        layout = self.layout
        props = context.scene.input_properties
        
        col = layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=False)
        col.alignment = 'CENTER'
        col.label(text="Substeps")
        col.prop(props, "substeps")
        
        row = layout.row()
        row.label(text="Constraint Iterations:", icon='LOOP_FORWARDS')
        
        col = layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=False)
        col.alignment = 'CENTER'
        col.label(text="Distance")
        col.prop(props, "distanceIterations")
        col.label(text="Volume")
        col.prop(props, "volumeIterations")
        col.label(text="Collision")
        col.prop(props, "collisionIterations")
#        layout.label(text=f"Current Value: {props.my_int:0}")

        row = layout.row()
        row.label(text="Distance Constraints:", icon='ARROW_LEFTRIGHT')
        
        col = layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=False)
        col.alignment = 'CENTER'
        col.label(text="Weight")
        col.prop(props, "distanceWeight")
#        col.label(text=f"Current Value: {props.my_float:.2f}")

        col.label(text="Stiffness")
        col.prop(props, "distanceStiffness")
        
        col.label(text="Damping")
        col.prop(props, "distanceDamping")
        
        row = layout.row()
        row.label(text="Volume Constraints:", icon='CUBE')
        
        col = layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=False)
        col.alignment = 'CENTER'
        col.label(text="Preservation")
        col.prop(props, "volumeStiffness")
        
        col.label(text="Damping")
        col.prop(props, "volumeDamping")
        
        row = layout.row()
        row.label(text="Forces:", icon='FORCE_FORCE')
        
        col = layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=False)
        col.alignment = 'CENTER'
        col.label(text="Collision Radius")
        col.prop(props, "collisionRadius")

        col.label(text="Friction")
        col.prop(props, "friction")
        
        row = layout.row()
        row.label(text="Pin Points:", icon='SNAP_MIDPOINT')
        
        col = layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=False)
        col.alignment = 'CENTER'
        col.label(text="Pin Group")
        col.prop(props, "pinGroup")
        
        row = layout.row()
        row.operator("object.reset_position_button")


class TetrahedralWorkshopButton(bpy.types.Operator):
    bl_idname = "object.tetrahedral_workshop_button"
    bl_label = "Tetrahedral Workshop"
    
    def execute(self, context):
        
        # Add object to physicsObjectsList, remove it if already there.
        if bpy.context.object.type == 'MESH':
            if bpy.context.object in objects:
                reset_position()
                objects.remove(bpy.context.object)
                
                bpy.utils.unregister_class(TetrahedralWorkshopPanel)
                bpy.app.handlers.animation_playback_pre.clear()
                bpy.app.handlers.animation_playback_post.clear()
                bpy.app.handlers.frame_change_pre.clear()
                bpy.app.handlers.render_pre.clear()
                bpy.app.handlers.render_init.clear()
                
                del bpy.types.Scene.input_properties
                
            elif bpy.context.object not in objects:
                objects.clear()
                objects.append(bpy.context.object)
                if not TetrahedralWorkshopPanel.is_registered:
                    bpy.utils.register_class(TetrahedralWorkshopPanel)
                if not ResetPositionButton.is_registered:
                    bpy.utils.register_class(ResetPositionButton)
                bpy.app.handlers.animation_playback_pre.clear()
                bpy.app.handlers.animation_playback_post.clear()
                bpy.app.handlers.animation_playback_pre.append(on_playback_start)
                bpy.app.handlers.animation_playback_post.append(on_playback_stop)
                bpy.app.handlers.render_pre.append(on_render)
                bpy.app.handlers.render_init.append(init_render)
                
                bpy.types.Scene.input_properties = bpy.props.PointerProperty(type=InputProperties)

                global props
                props = bpy.context.scene.input_properties
                
                save_position()
                init_physics()
                
        return {'FINISHED'}

class TetrahedralWorkshop(bpy.types.Panel):
    bl_label = "Tetrahedral Workshop"
    bl_idname = "OBJECT_PT_TETRAHEDRALWORKSHOP"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "physics"
    bl_options = {'HIDE_HEADER'}
    
    bpy.utils.register_class(TetrahedralWorkshopButton)

    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        row.operator("object.tetrahedral_workshop_button", icon="MESH_DATA")

def register():
    bpy.app.handlers.depsgraph_update_post.clear()
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.animation_playback_pre.clear()
    bpy.app.handlers.animation_playback_post.clear()
    
    
    bpy.utils.register_class(TetrahedralWorkshop)
    bpy.app.handlers.depsgraph_update_post.append(on_click)
    
    bpy.utils.register_class(InputProperties)
#    bpy.types.Scene.input_properties = bpy.props.PointerProperty(type=InputProperties)

def unregister():

    bpy.utils.unregister_class(TetrahedralWorkshop)
    bpy.utils.unregister_class(InputProperties)
    
    bpy.app.handlers.depsgraph_update_post.clear()
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.animation_playback_pre.clear()
    bpy.app.handlers.animation_playback_post.clear()
    
#    del bpy.types.Scene.input_properties
    
if __name__ == "__main__":
    register()