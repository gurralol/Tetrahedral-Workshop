bl_info = {
    "name": "Tetrahedral Workshop",
    "blender": (2, 80, 0),
    "category": "Physics",
}

import bpy
import mathutils
from mathutils.bvhtree import BVHTree
import math
import bmesh
import numpy as np
import time

#//////////////////////////////////////////////////////////////////////////////////

SoftBodyArray = np.empty(5, dtype=object)

#//////////////////////////////////////////////////////////////////////////////////

def get_tet_volume(obj, arr, nr):
    id0 = arr[nr][0]
    id1 = arr[nr][1]
    id2 = arr[nr][2]
    id3 = arr[nr][3]
    
    v0 = obj.data.vertices[id1].co.copy() - obj.data.vertices[id0].co.copy()
    v1 = obj.data.vertices[id2].co.copy() - obj.data.vertices[id0].co.copy()
    v2 = obj.data.vertices[id3].co.copy() - obj.data.vertices[id0].co.copy()
    cross = np.cross(v0, v1)
    
    return np.dot(cross, v2) / 6.0

def vector_length_squared(vector):
    a0 = vector[0]
    a1 = vector[1]
    a2 = vector[2]
    return a0 * a0 + a1 * a1 + a2 * a2

def get_animated_bmesh(obj):
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
    return bm

#//////////////////////////////////////////////////////////////////////////////////

class SoftBody:
    def __init__(self, obj):
        #-------------------------------------------------------------------
        # Initialize object
        self.obj = obj
        # Initialize rest position array
        self.restPos = np.empty(len(obj.data.vertices), dtype=mathutils.Vector)
        # Initialize current position array
        self.currentPos = np.empty(len(obj.data.vertices), dtype=mathutils.Vector)
        # Initialize previous position array
        self.previousPos = np.empty(len(obj.data.vertices), dtype=mathutils.Vector)
        # Initialize tet ID:s array
        quads = 0
        for i in range (len(self.obj.data.polygons)):
            if len(self.obj.data.polygons[i].vertices) == 4:
                quads += 1
        self.tetIds = np.empty((quads, 4), dtype=int)
        # Initialize tet rest volumes array
        #self.tetVolumes = np.empty(quads, dtype=float)
        # Initialize rest volume array
        self.restVol = np.empty(quads, float)
        # Initialize inverse mass array
        self.invMass = np.empty(len(self.obj.data.vertices), dtype=float)
        # Initialize edge ID:s array
        self.edgeIds = np.empty(((len(self.obj.data.edges)), 2), dtype=int)
        # Initialize edge Lengths array
        self.edgeLengths = np.empty((len(self.obj.data.edges)), dtype=float)
        # Initialize volume ID order array
        self.volIdOrder = np.array([[1,3,2], [0,2,3], [0,3,1], [0,1,2]])
        # Initialize grads array
        self.grads = np.empty(4, dtype=mathutils.Vector)
        #-------------------------------------------------------------------
        # Store rest position
        for i in range (len(self.obj.data.vertices)):
            self.restPos[i] = self.obj.data.vertices[i].co.copy()
        # Store tet ID:s
        for i in range (len(obj.data.polygons)):
            if len(self.obj.data.polygons[i].vertices) == 4:
                id0, id1, id2, id3 = self.obj.data.polygons[i].vertices
                self.tetIds[i] = (id0, id1, id2, id3)
        # Store rest volume
        for i in range (len(self.tetIds)):
            self.restVol[i] = get_tet_volume(self.obj, self.tetIds, i)
        # Store inverse mass
        invMassTemp = 0.0
        for i in range (quads):
            invMass = self.restVol[i]
            pInvMass = 1.0 / (invMassTemp / 4.0) if invMassTemp < 0.0 else 0.0
            for j in self.obj.data.polygons[i].vertices:
                self.invMass[j] += invMassTemp
        # Store rest distance
        for i in range (len(self.obj.data.edges)):
            id0, id1 = self.obj.data.edges[i].vertices
            pos0 = self.obj.data.vertices[id0].co.copy()
            pos1 = self.obj.data.vertices[id1].co.copy()
            distance = (pos0 - pos1).length
            
            self.edgeIds[i] = (id0, id1)
            self.edgeLengths[i] = distance
        #-------------------------------------------------------------------
        
    def reset_position(self):
        for i in range (len(self.obj.data.vertices)):
            self.obj.data.vertices[i].co = self.restPos[i]
            self.currentPos[i] = None
            self.previousPos[i] = None
        return
    
    def pre_solve(self, sdt, gravity):
        gravity = gravity * sdt
        
        # Collect objects with collision enabled.
        bmCollisions = bmesh.new()
        for i in bpy.data.objects:
            if i.type == 'MESH':
                for j in i.modifiers:
                    if j.type == 'COLLISION':
                        bmCollisions = get_animated_bmesh(i)
        bvhTreeCollisions = BVHTree.FromBMesh(bmCollisions)
        
        # Create bmesh with world coordinates.
        bmObj = bmesh.new()
        bmObj.from_object(self.obj, bpy.context.evaluated_depsgraph_get())
        bmObj.verts.ensure_lookup_table()
        
        for j in range (len(self.obj.data.vertices)):
            self.currentPos[j] = bmObj.verts[j].co.copy()
            if self.previousPos[j] == None:
                self.previousPos[j] = self.currentPos[j]
                continue
            velocity = self.currentPos[j] - self.previousPos[j]
            velocity = velocity * (1.0 / sdt)
            displacement = velocity + gravity
            displacement = displacement * sdt
            if self.obj.tet_properties.collisionIterations == 0:
                self.obj.data.vertices[j].co += displacement
            
            for its in range(self.obj.tet_properties.collisionIterations):
                displacementIterative = displacement / self.obj.tet_properties.collisionIterations
                self.obj.data.vertices[j].co += displacementIterative
                
                worldPos = bmObj.verts[j].co.copy()
                location, normal, index, distance = bvhTreeCollisions.find_nearest(worldPos)
                if distance != None and distance < self.obj.tet_properties.collisionRadius:
                    # Collision
                    pushDirection = normal.normalized() * (self.obj.tet_properties.collisionRadius - distance)
                    self.obj.data.vertices[j].co = (worldPos + pushDirection)
                    # Friction
                    velocityAtVertex = velocity
                    normalComponent = velocityAtVertex.project(normal)
                    tangentialComponent = velocityAtVertex - normalComponent
                    tangentialComponent *= self.obj.tet_properties.friction
                    velocity = normalComponent + tangentialComponent 
                    velocity = velocity / self.obj.tet_properties.collisionIterations
                    velocity = velocity * sdt
                    self.obj.data.vertices[j].co -= velocity
                    
            # Pin vertices to animation
            try:
                if self.obj.vertex_groups[self.obj.tet_properties.pinGroup].weight(j) > 0.9:
                    self.obj.data.vertices[j].co = self.restPos[j]
            except:
                pass
        bmCollisions.free()
        bmObj.free()
        return
        
    def solve(self, sdt):
        for i in range(self.obj.tet_properties.distanceIterations):
            self.solve_edges(sdt)
        for i in range(self.obj.tet_properties.volumeIterations):
            self.solve_volumes(sdt)
        return
    
    def post_solve(self, sdt):
        for i in range (len(self.obj.data.vertices)):
            self.previousPos[i] = self.currentPos[i]
        return
    
    def simulate(self, dt, gravity):
        sdt = dt / self.obj.tet_properties.substeps
        for i in range(self.obj.tet_properties.substeps):
            self.pre_solve(sdt, gravity)
            self.solve(sdt)
            self.post_solve(sdt)
        return
    
    def solve_edges(self, sdt):
        alpha = self.obj.tet_properties.distanceStiffness / sdt / sdt
        
        for its in range(self.obj.tet_properties.distanceIterations):
            bm = bmesh.new()
            bm.from_object(self.obj, bpy.context.evaluated_depsgraph_get())
            bm.verts.ensure_lookup_table()

            for i in range (len(self.edgeLengths)):
                id0 = self.edgeIds[i][0]
                id1 = self.edgeIds[i][1]
                pos_i = bm.verts[id0].co.copy()
                pos_j = bm.verts[id1].co.copy()
                w0 = 0.5
                w1 = 0.5
                w = w0 + w1
                if w == 0.0:
                    continue
            
                vector = (pos_i - pos_j)
                currentDistance = (pos_i - pos_j).length
                if currentDistance == 0:
                    continue
                vector *= 1.0 / currentDistance
                distance = self.edgeLengths[i]
                direction = (pos_i - pos_j) / currentDistance
                C = currentDistance - distance
                s = -C / (w * alpha)
                
                displacement = vector * ((s * w0) * self.obj.tet_properties.distanceMass)
                displacement *= self.obj.tet_properties.distanceDamping
                self.obj.data.vertices[id0].co += displacement
                self.obj.data.vertices[id1].co -= displacement
                
    def solve_volumes(self, sdt):
        alpha = self.obj.tet_properties.volumeStiffness / sdt / sdt
        for its in range(self.obj.tet_properties.volumeIterations):
            bm = bmesh.new()
            bm.from_object(self.obj, bpy.context.evaluated_depsgraph_get())
            bm.verts.ensure_lookup_table()
            
            for i in range (len(self.tetIds)):
                w = 0.0
                for j in range(4):
                    id0 = self.tetIds[i][self.volIdOrder[j][0]]
                    id1 = self.tetIds[i][self.volIdOrder[j][1]]
                    id2 = self.tetIds[i][self.volIdOrder[j][2]]
                    temp0 = bm.verts[id1].co.copy() - bm.verts[id0].co.copy()
                    temp1 = bm.verts[id2].co.copy() - bm.verts[id0].co.copy()
                    #cross = temp0.cross(temp1)
                    crossV = np.cross(temp0, temp1)
                    cross = mathutils.Vector((crossV[0], crossV[1], crossV[2]))
                    if cross.length == 0:
                        continue
                    cross /= 6
                    self.grads[j] = cross
                    w += -1.0 * vector_length_squared(cross)
                
                vol = get_tet_volume(self.obj, self.tetIds, i)
                rVol = self.restVol[i]
                C = vol - rVol
                s = -C / (w * alpha)
                
                for j in range(4):
                    id = self.tetIds[i][j]
                    displacement = self.grads[j]  * (s * (w / 4))
                    displacement *= self.obj.tet_properties.volumeDamping
                    self.obj.data.vertices[id].co += displacement
        return

#//////////////////////////////////////////////////////////////////////////////////

def simulate(scene):
    dt = bpy.context.scene.render.fps_base / bpy.context.scene.render.fps
    gravity = bpy.context.scene.gravity.copy()
    for i in range (len(SoftBodyArray)):
        if SoftBodyArray[i] != None:
            SoftBodyArray[i].simulate(dt, gravity)

def on_playback_start(scene):
    bpy.app.handlers.frame_change_pre.append(simulate)
    return

def on_playback_stop(scene):
    bpy.app.handlers.frame_change_pre.remove(simulate)
    return

def on_render_init(scene):
    return

def on_render_frame(scene):
    return

def reset_positions(scene):
    for i in range (len(SoftBodyArray)):
        if SoftBodyArray[i] != None:
            if bpy.context.scene.frame_current == 1:
                SoftBodyArray[i].reset_position()

#//////////////////////////////////////////////////////////////////////////////////

class TetProperties(bpy.types.PropertyGroup):
        substeps: bpy.props.IntProperty(
            name="",
            description="",
            default=6,
            min=0,
            max=100,
            step=1,
        )
        distanceIterations: bpy.props.IntProperty(
            name="",
            description="",
            default=1,
            min=0,
            max=100,
            step=1,
        )
        volumeIterations: bpy.props.IntProperty(
            name="",
            description="",
            default=1,
            min=0,
            max=100,
            step=1,
        )
        collisionIterations: bpy.props.IntProperty(
            name="",
            description="",
            default=1,
            min=0,
            max=100,
            step=1,
        )
        
        distanceMass: bpy.props.FloatProperty(
            name="",
            description="",
            default=1.0,
            min=0.0,
            max=100.0,
            step=10,
            precision=2
        )
        distanceStiffness: bpy.props.FloatProperty(
            name="",
            description="",
            default=1.0,
            min=0.0,
            max=100.0,
            step=10,
            precision=2
        )
        distanceDamping: bpy.props.FloatProperty(
            name="",
            description="",
            default=1.0,
            min=0.0,
            max=100.0,
            step=10,
            precision=2
        )
        
        volumeStiffness: bpy.props.FloatProperty(
            name="",
            description="",
            default=1.0,
            min=0.0,
            max=100.0,
            step=10,
            precision=2
        )
        volumeDamping: bpy.props.FloatProperty(
            name="",
            description="",
            default=1.0,
            min=0.0,
            max=100.0,
            step=10,
            precision=2
        )
        
        collisionRadius: bpy.props.FloatProperty(
            name="",
            description="",
            default=1.0,
            min=0.0,
            max=100.0,
            step=10,
            precision=2
        )
        friction: bpy.props.FloatProperty(
            name="",
            description="",
            default=1.0,
            min=0.0,
            max=100.0,
            step=10,
            precision=2
        )
        
        pinGroup: bpy.props.StringProperty(
            name="",
            description="",
            default=""
        )
        
#//////////////////////////////////////////////////////////////////////////////////

# Main class
class TetrahedralWorkshop(bpy.types.Panel):
    bl_label = "TetrahedralWorkshop"
    bl_idname = "OBJECT_PT_TETRAHEDRALWORKSHOP"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "physics"
    bl_options = {'HIDE_HEADER'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("object.tetrahedral_workshop_button", icon="MESH_DATA")
    
    # Main button
    class TetrahedralWorkshopButton(bpy.types.Operator):
        bl_idname = "object.tetrahedral_workshop_button"
        bl_label = "Tetrahedral Workshop"
        
        # If the object is in SoftBodyArray, clear its properties and remove it.
        # If the object isn't in SoftBodyArray, add it to it.
        def execute(self, context):
            for i in range (len(SoftBodyArray)):
                if SoftBodyArray[i] is not None:
                    if SoftBodyArray[i].obj == bpy.context.object:
                        for propName, prop in SoftBodyArray[i].obj.tet_properties.bl_rna.properties.items():
                            SoftBodyArray[i].obj.tet_properties.property_unset(propName)
                        SoftBodyArray[i] = None
                        return {'FINISHED'}
            
            for i in range (len(SoftBodyArray)):
                if SoftBodyArray[i] is None:
                    SoftBodyArray[i] = SoftBody(bpy.context.object)
                    return {'FINISHED'}
            
            return {'FINISHED'}
    
    # Main panel
    class TetrahedralWorkshopPanel(bpy.types.Panel):
        bl_label = "Tetrahedral Workshop"
        bl_idname = "OBJECT_PT_TETRAHEDRALWORKSHOPPANEL"
        bl_space_type = 'PROPERTIES'
        bl_region_type = 'WINDOW'
        bl_context = "physics"
        
        # Only show panel if the selected object is in the SoftBodyArray.
        @classmethod
        def poll(cls, context):
            for i in range (len(SoftBodyArray)):
                if SoftBodyArray[i] is not None:
                    if SoftBodyArray[i].obj == bpy.context.object:
                        return True
            return False   
                
        def draw(self, context):
            layout = self.layout
            props = context.object.tet_properties
            
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
            
            row = layout.row()
            row.label(text="Distance Constraints", icon='ARROW_LEFTRIGHT')
            col = layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=False)
            col.alignment = 'CENTER'
            col.label(text="Mass")
            col.prop(props, "distanceMass")
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
        
    bpy.utils.register_class(TetrahedralWorkshopButton)
    bpy.utils.register_class(TetrahedralWorkshopPanel)
    bpy.utils.register_class(TetProperties)
    bpy.types.Object.tet_properties = bpy.props.PointerProperty(type=TetProperties)

#//////////////////////////////////////////////////////////////////////////////////

def register():
    # Clear handlers when reloading script...
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.animation_playback_pre.clear()
    bpy.app.handlers.animation_playback_post.clear()
    
    bpy.utils.register_class(TetrahedralWorkshop)
    bpy.app.handlers.animation_playback_pre.append(on_playback_start)
    bpy.app.handlers.animation_playback_post.append(on_playback_stop)
    bpy.app.handlers.frame_change_pre.append(reset_positions)

def unregister():
    bpy.utils.unregister_class(TetrahedralWorkshop)
    
    # Should only remove the specific functions instead of clear...
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.animation_playback_pre.clear()
    bpy.app.handlers.animation_playback_post.clear()

if __name__ == "__main__":
    register()