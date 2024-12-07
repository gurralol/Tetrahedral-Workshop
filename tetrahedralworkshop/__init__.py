bl_info = {
    "name": "Tetrahedral Workshop",
    "blender": (2, 80, 0),
    "category": "Physics",
    "description": "Tetrahedral soft-body simulator.",
    "doc_url": "https://github.com/gurralol/Tetrahedral-Workshop",
}

import bpy
import mathutils
from mathutils.bvhtree import BVHTree
import math
import bmesh

#//////////////////////////////////////////////////////////////////////////////////

SoftBodyList = []

#//////////////////////////////////////////////////////////////////////////////////

def get_tet_volume(obj, arr, nr):
    id0 = arr[nr][0]
    id1 = arr[nr][1]
    id2 = arr[nr][2]
    id3 = arr[nr][3]
    v0 = obj.data.vertices[id1].co.copy() - obj.data.vertices[id0].co.copy()
    v1 = obj.data.vertices[id2].co.copy() - obj.data.vertices[id0].co.copy()
    v2 = obj.data.vertices[id3].co.copy() - obj.data.vertices[id0].co.copy()
    cross = v0.cross(v1)
    return cross.dot(v2) / 6.0

def get_animated_bmesh(bm, obj):
    bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
    return bm

#//////////////////////////////////////////////////////////////////////////////////

class SoftBody:
    def __init__(self, obj):
        #-------------------------------------------------------------------
        # Initialize object
        self.obj = obj
        # Initialize verts array
        self.verts = []
        # Initialize rest position array
        self.restPos = tuple(self.obj.data.vertices[i].co.copy() for i in range(len(self.obj.data.vertices)))
        # Initialize current position array
        self.currentPos = []
        # Initialize previous position array
        self.previousPos = []
        # Initialize tet ID:s array
        self.tetIds = []
        # Initialize rest volume array
        self.restVol = []
        # Initialize inverse mass array
        self.invMass = []
        # Initialize edge ID:s array
        self.edgeIds = []
        # Initialize edge Lengths array
        self.edgeLengths = []
        # Initialize volume ID order array
        self.volIdOrder = [[1,3,2], [0,2,3], [0,3,1], [0,1,2]]
        # Initialize grads array
        self.grads = [[mathutils.Vector((0.0, 0.0, 0.0))], [mathutils.Vector((0.0, 0.0, 0.0))], [mathutils.Vector((0.0, 0.0, 0.0))], [mathutils.Vector((0.0, 0.0, 0.0))], ]
        # Initialize edge count array
        self.edgeCount = len(self.obj.data.edges)
        # Initialize vert count array
        self.vertCount = len(self.obj.data.vertices)
        # Initialize Obj Bmesh
        self.bmObj = None
        # Initialize BVHTree array
        self.bvhTreeCollisions = None
        #-------------------------------------------------------------------
        # Store rest position
        for i in range (len(self.obj.data.vertices)):
            #self.restPos.append(self.obj.data.vertices[i].co.copy())
            self.currentPos.append(self.obj.data.vertices[i].co.copy())
            self.previousPos.append(self.obj.data.vertices[i].co.copy())
            self.verts.append(self.obj.data.vertices[i].co.copy())
        # Store tet ID:s
        for i in range (len(obj.data.polygons)):
            if len(self.obj.data.polygons[i].vertices) == 4:
                id0, id1, id2, id3 = self.obj.data.polygons[i].vertices
                self.tetIds.append((id0, id1, id2, id3))
        # Store rest volume
        for i in range (len(self.tetIds)):
            self.restVol.append(get_tet_volume(self.obj, self.tetIds, i))
        # Store inverse mass
        for i in range (len(obj.data.polygons)):
            if len(self.obj.data.polygons[i].vertices) == 4:
                invMass = self.restVol[i]
        # Store rest distance
        for i in range (len(self.obj.data.edges)):
            id0, id1 = self.obj.data.edges[i].vertices
            pos0 = self.obj.data.vertices[id0].co.copy()
            pos1 = self.obj.data.vertices[id1].co.copy()
            distance = (pos0 - pos1).length
            
            self.edgeIds.append((id0, id1))
            self.edgeLengths.append(distance)
        #-------------------------------------------------------------------
    
    def reset_position(self):
        for i in range (len(self.obj.data.vertices)):
            self.obj.data.vertices[i].co = self.restPos[i].copy()
            self.verts[i] = self.restPos[i].copy()
            self.currentPos[i] = self.restPos[i].copy()
            self.previousPos[i] = self.restPos[i].copy()
        return
    
    def populate_bmObj(self):
        self.bmObj = bmesh.new()
        self.bmObj.from_object(self.obj, bpy.context.evaluated_depsgraph_get())
        self.bmObj.verts.ensure_lookup_table()
    
    def crazyspace(self):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        scene = bpy.context.scene
        self.obj.crazyspace_eval(depsgraph, scene)
    
    def populate_bmCollisions(self):
        bmCollisions = bmesh.new()
        for i in bpy.data.objects:
            if i.type == 'MESH':
                for j in i.modifiers:
                    if j.type == 'COLLISION':
                        bmCollisions = get_animated_bmesh(bmCollisions, i)
        self.bvhTreeCollisions = BVHTree.FromBMesh(bmCollisions)
        bmCollisions.free()
        
    def simulate(self, dt, gravity):
        sdt = dt / self.obj.tet_properties.substeps
        gravity = (gravity * sdt)
        
        self.populate_bmObj()
        
        for i in range(self.vertCount):
            self.verts[i] = self.bmObj.verts[i].co.copy()
 
        #self.crazyspace()
        self.populate_bmCollisions()
        for i in range(self.obj.tet_properties.substeps):
            self.pre_solve(sdt, gravity)
            self.solve(sdt)
            self.post_solve()
        
        for i in range(self.vertCount):
            try:
                if self.obj.vertex_groups[self.obj.tet_properties.pinGroup].weight(i) > 0.9:
                    self.verts[i] = self.restPos[i].copy()
            except:
                pass
        
        flat_list = []
        for row in self.verts:
            flat_list.extend(row)
        self.obj.data.vertices.foreach_set("co", flat_list)
        
        self.bmObj.free()
        #self.obj.crazyspace_eval_clear()
    
    def pre_solve(self, sdt, gravity):
        for j in range(self.vertCount):
            try:
                if self.obj.vertex_groups[self.obj.tet_properties.pinGroup].weight(j) == 1.0:
                    continue
            except:
                pass
            self.currentPos[j] = self.verts[j].copy()
            velocity = self.currentPos[j].copy() - self.previousPos[j].copy()
            if velocity.length > 0.000000000000001:
                velocity = velocity * (1.0 / sdt)
            displacement = velocity + gravity
            displacement = displacement * sdt
            if displacement.length > 100000.0 or displacement.length < 0.000000000000001:
                continue
            if self.obj.tet_properties.collisionIterations == 0:
                self.verts[j] += displacement
            else:
                self.collisions(sdt, j, displacement, velocity)
    
    def collisions(self, sdt, j, displacement, velocity):
        for its in range(self.obj.tet_properties.collisionIterations):
            displacementIterative = displacement / self.obj.tet_properties.collisionIterations
            if displacementIterative.length > 100000.0 or displacementIterative.length < 0.000000000000001:
                continue
            self.verts[j] += displacementIterative
            location, bvhNormal, index, bvhDistance = self.bvhTreeCollisions.find_nearest(self.verts[j].copy())
            if bvhDistance != None and bvhDistance < self.obj.tet_properties.collisionRadius:
                pushDirection = bvhNormal.normalized() * (self.obj.tet_properties.collisionRadius - bvhDistance)
                self.verts[j] = (self.verts[j].copy() + pushDirection)
                normalComponent = velocity.project(bvhNormal)
                tangentialComponent = velocity - normalComponent
                tangentialComponent *= self.obj.tet_properties.friction
                velocity = normalComponent + tangentialComponent
                if velocity.length > 100000.0 or velocity.length < 0.000000000000001:
                    continue
                velocity = velocity / self.obj.tet_properties.collisionIterations
                velocity = velocity * sdt
                self.verts[j] -= velocity
            
    def solve(self, sdt):
        for i in range(self.obj.tet_properties.distanceIterations):
            self.solve_edges(sdt)
        for i in range(self.obj.tet_properties.volumeIterations):
            self.solve_volumes(sdt)
        #for i in range(self.obj.tet_properties.pinIterations):
            #self.solve_pin(sdt)

    def solve_edges(self, sdt):
        distanceAlpha = 1 / (self.obj.tet_properties.distanceStiffness * float(self.obj.tet_properties.distanceStiffnessE)) / sdt / sdt
        distanceMass = 1 / (self.obj.tet_properties.distanceMass * float(self.obj.tet_properties.distanceMassE))
        distanceDamping = 1 / (self.obj.tet_properties.distanceDamping * float(self.obj.tet_properties.distanceDampingE))
        
        for i in range(len(self.edgeIds)):
            try:
                if self.obj.vertex_groups[self.obj.tet_properties.pinGroup].weight(i) == 1.0:
                    continue
            except:
                pass
            id0 = self.edgeIds[i][0]
            id1 = self.edgeIds[i][1]
            pos_i = self.verts[id0].copy()
            pos_j = self.verts[id1].copy()
            vector = pos_i - pos_j
            currentDistance = vector.length
            if currentDistance < 0.000000000000001:
                continue
            vector *= 1.0 / currentDistance
            distance = self.edgeLengths[i]
            direction = vector / currentDistance
            C = currentDistance - distance
            s = -C / distanceAlpha
            displacement = (vector * (s * distanceMass)) * distanceDamping
            if displacement.length > 100000.0 or displacement.length < 0.000000000000001:
                continue
            self.verts[id0] += displacement
            self.verts[id1] -= displacement

    def solve_volumes(self, sdt):
        volumeAlpha = 1 / (self.obj.tet_properties.volumeStiffness * float(self.obj.tet_properties.volumeStiffnessE)) / sdt / sdt
        volumeDamping = 1 / (self.obj.tet_properties.volumeDamping * float(self.obj.tet_properties.volumeDampingE))
        
        for i in range(len(self.tetIds)):
            try:
                if self.obj.vertex_groups[self.obj.tet_properties.pinGroup].weight(i) == 1.0:
                    continue
            except:
                pass
            w = self.get_tet_weight(i)
            vol = self.get_tet_volume(i)
            rVol = self.restVol[i]
            C = vol - rVol
            if C > 1.0:
                continue
            s = -C / (w * volumeAlpha)
            self.tet_displace(i, s, w, volumeDamping)
                
    def get_tet_weight(self, i):
        w = 0.0
        for j in range(4):
            id0 = self.tetIds[i][self.volIdOrder[j][0]]
            id1 = self.tetIds[i][self.volIdOrder[j][1]]
            id2 = self.tetIds[i][self.volIdOrder[j][2]]
            temp0 = self.verts[id1].copy() - self.verts[id0].copy()
            temp1 = self.verts[id2].copy() - self.verts[id0].copy()
            cross = temp0.cross(temp1)
            crossLength = cross.length
            if crossLength < 0.000000000000001:
                continue
            cross /= 6
            self.grads[j] = cross
            w += -1.0 * crossLength
        return w
    
    def get_tet_volume(self, i):
        id0 = self.tetIds[i][0]
        id1 = self.tetIds[i][1]
        id2 = self.tetIds[i][2]
        id3 = self.tetIds[i][3]
        v0 = self.verts[id1].copy() - self.verts[id0].copy()
        v1 = self.verts[id2].copy() - self.verts[id0].copy()
        v2 = self.verts[id3].copy() - self.verts[id0].copy()
        cross = v0.cross(v1)
        if cross.length > 0.000000000000001:
            result = cross.dot(v2) / 6.0
        else:
            result = 0.0
        return result
    
    def tet_displace(self, i, s, w, volumeDamping):
        for j in range(4):
            id = self.tetIds[i][j]
            displacement = (self.grads[j]  * (s * (w / 4))) * volumeDamping
            if displacement.length > 100000.0 or displacement.length < 0.000000000000001:
                continue
            self.verts[id] += displacement
            
    def solve_pin(self, sdt):
        for i in range(self.vertCount):
            stiffness = self.obj.vertex_groups[self.obj.tet_properties.pinGroup].weight(i)
        return
    
    def post_solve(self):
        self.previousPos[:] = self.currentPos

#//////////////////////////////////////////////////////////////////////////////////

def simulate(scene):
    dt = bpy.context.scene.render.fps_base / bpy.context.scene.render.fps
    gravity = bpy.context.scene.gravity.copy()
    for i in SoftBodyList:
        if i != None:
            i.simulate(dt, gravity)

def on_playback_start(scene):
    bpy.app.handlers.frame_change_pre.append(simulate)
    return

def on_playback_stop(scene):
    bpy.app.handlers.frame_change_pre.remove(simulate)
    return

def on_render_pre(scene):
    bpy.context.scene.render.use_lock_interface = True
    dt = bpy.context.scene.render.fps_base / bpy.context.scene.render.fps
    gravity = bpy.context.scene.gravity.copy()
    for i in SoftBodyList:
        if i != None:
            i.simulate(dt, gravity)

def reset_positions(scene):
    for i in range (len(SoftBodyList)):
        if SoftBodyList[i] != None:
            if bpy.context.scene.frame_current == bpy.context.scene.frame_start:
                SoftBodyList[i].reset_position()

#//////////////////////////////////////////////////////////////////////////////////

class TetProperties(bpy.types.PropertyGroup):
        substeps: bpy.props.IntProperty(
            name="",
            description="",
            default=3,
            min=0,
            max=1000,
            step=1
        )
        distanceIterations: bpy.props.IntProperty(
            name="",
            description="",
            default=1,
            min=0,
            max=1000,
            step=1
        )
        volumeIterations: bpy.props.IntProperty(
            name="",
            description="",
            default=1,
            min=0,
            max=1000,
            step=1
        )
        collisionIterations: bpy.props.IntProperty(
            name="",
            description="",
            default=3,
            min=0,
            max=1000,
            step=1
        )
        pinIterations: bpy.props.IntProperty(
            name="",
            description="",
            default=1,
            min=0,
            max=1000,
            step=1
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
        distanceMassE: bpy.props.EnumProperty(
            name="",
            items=[
                ('10000000000', "1e+10", ""),
                ('1000000000', "1e+9", ""),
                ('100000000', "1e+8", ""),
                ('10000000', "1e+7", ""),
                ('1000000', "1e+6", ""),
                ('100000', "1e+5", ""),
                ('10000', "1e+4", ""),
                ('1000', "1000", ""),
                ('100', "100", ""),
                ('10', "10", ""),
                ('1', "1", ""),
                ('0.1', "0.1", ""),
                ('0.01', "0.01", ""),
                ('0.001', "0.001", ""),
                ('0.0001', "1e-4", ""),
                ('0.00001', "1e-5", ""),
                ('0.000001', "1e-6", ""),
                ('0.0000001', "1e-7", ""),
                ('0.00000001', "1e-8", ""),
                ('0.000000001', "1e-9", ""),
                ('0.0000000001', "1e-10", "")
            ],
            default='1'
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
        distanceStiffnessE: bpy.props.EnumProperty(
            name="",
            items=[
                ('10000000000', "1e+10", ""),
                ('1000000000', "1e+9", ""),
                ('100000000', "1e+8", ""),
                ('10000000', "1e+7", ""),
                ('1000000', "1e+6", ""),
                ('100000', "1e+5", ""),
                ('10000', "1e+4", ""),
                ('1000', "1000", ""),
                ('100', "100", ""),
                ('10', "10", ""),
                ('1', "1", ""),
                ('0.1', "0.1", ""),
                ('0.01', "0.01", ""),
                ('0.001', "0.001", ""),
                ('0.0001', "1e-4", ""),
                ('0.00001', "1e-5", ""),
                ('0.000001', "1e-6", ""),
                ('0.0000001', "1e-7", ""),
                ('0.00000001', "1e-8", ""),
                ('0.000000001', "1e-9", ""),
                ('0.0000000001', "1e-10", "")
            ],
            default='1000'
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
        distanceDampingE: bpy.props.EnumProperty(
            name="",
            items=[
                ('10000000000', "1e+10", ""),
                ('1000000000', "1e+9", ""),
                ('100000000', "1e+8", ""),
                ('10000000', "1e+7", ""),
                ('1000000', "1e+6", ""),
                ('100000', "1e+5", ""),
                ('10000', "1e+4", ""),
                ('1000', "1000", ""),
                ('100', "100", ""),
                ('10', "10", ""),
                ('1', "1", ""),
                ('0.1', "0.1", ""),
                ('0.01', "0.01", ""),
                ('0.001', "0.001", ""),
                ('0.0001', "1e-4", ""),
                ('0.00001', "1e-5", ""),
                ('0.000001', "1e-6", ""),
                ('0.0000001', "1e-7", ""),
                ('0.00000001', "1e-8", ""),
                ('0.000000001', "1e-9", ""),
                ('0.0000000001', "1e-10", "")
            ],
            default='1'
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
        volumeStiffnessE: bpy.props.EnumProperty(
            name="",
            items=[
                ('10000000000', "1e+10", ""),
                ('1000000000', "1e+9", ""),
                ('100000000', "1e+8", ""),
                ('10000000', "1e+7", ""),
                ('1000000', "1e+6", ""),
                ('100000', "1e+5", ""),
                ('10000', "1e+4", ""),
                ('1000', "1000", ""),
                ('100', "100", ""),
                ('10', "10", ""),
                ('1', "1", ""),
                ('0.1', "0.1", ""),
                ('0.01', "0.01", ""),
                ('0.001', "0.001", ""),
                ('0.0001', "1e-4", ""),
                ('0.00001', "1e-5", ""),
                ('0.000001', "1e-6", ""),
                ('0.0000001', "1e-7", ""),
                ('0.00000001', "1e-8", ""),
                ('0.000000001', "1e-9", ""),
                ('0.0000000001', "1e-10", "")
            ],
            default='100000'
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
        volumeDampingE: bpy.props.EnumProperty(
            name="",
            items=[
                ('10000000000', "1e+10", ""),
                ('1000000000', "1e+9", ""),
                ('100000000', "1e+8", ""),
                ('10000000', "1e+7", ""),
                ('1000000', "1e+6", ""),
                ('100000', "1e+5", ""),
                ('10000', "1e+4", ""),
                ('1000', "1000", ""),
                ('100', "100", ""),
                ('10', "10", ""),
                ('1', "1", ""),
                ('0.1', "0.1", ""),
                ('0.01', "0.01", ""),
                ('0.001', "0.001", ""),
                ('0.0001', "1e-4", ""),
                ('0.00001', "1e-5", ""),
                ('0.000001', "1e-6", ""),
                ('0.0000001', "1e-7", ""),
                ('0.00000001', "1e-8", ""),
                ('0.000000001', "1e-9", ""),
                ('0.0000000001', "1e-10", "")
            ],
            default='1'
        )
        
        collisionRadius: bpy.props.FloatProperty(
            name="",
            description="",
            default=0.1,
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
        
        cache: bpy.props.EnumProperty(
            name="",
            items=[
                ('None', "", ""),
                ('OPT0', "Option 1", ""),
                ('OPT1', "Option 2", ""),
            ],
            default='None'
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
        for i in SoftBodyList:
            if i.obj not in bpy.data.objects.values():
                SoftBodyList.remove(i)
        return context.active_object.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("object.tetrahedral_workshop_button", icon="MESH_DATA")
    
    # Main button
    class TetrahedralWorkshopButton(bpy.types.Operator):
        bl_idname = "object.tetrahedral_workshop_button"
        bl_label = "Tetrahedral Workshop"

        def execute(self, context):
            for i in range(len(SoftBodyList)):
                if SoftBodyList[i].obj == bpy.context.object:
                    SoftBodyList[i].reset_position()
                    for propName, prop in SoftBodyList[i].obj.tet_properties.bl_rna.properties.items():
                            SoftBodyList[i].obj.tet_properties.property_unset(propName)
                    SoftBodyList.remove(SoftBodyList[i])
                    return {'FINISHED'}
            if len(SoftBodyList) == 0:
                SoftBodyList.append(SoftBody(bpy.context.object))
                return {'FINISHED'}
            for i in SoftBodyList:
                if i.obj != bpy.context.object:
                    SoftBodyList.append(SoftBody(bpy.context.object))
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
            for i in SoftBodyList:
                if i != None:
                    if bpy.context.object == i.obj:
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
            col = layout.grid_flow(row_major=True, columns=3, even_columns=True, even_rows=True, align=False)
            col.alignment = 'CENTER'
            col.label(text="Mass")
            col.prop(props, "distanceMass")
            col.prop(props, "distanceMassE")
            col.label(text="Stiffness")
            col.prop(props, "distanceStiffness")
            col.prop(props, "distanceStiffnessE")
            col.label(text="Damping")
            col.prop(props, "distanceDamping")
            col.prop(props, "distanceDampingE")
            
            row = layout.row()
            row.label(text="Volume Constraints:", icon='CUBE')
            col = layout.grid_flow(row_major=True, columns=3, even_columns=True, even_rows=False, align=False)
            col.alignment = 'CENTER'
            col.label(text="Preservation")
            col.prop(props, "volumeStiffness")
            col.prop(props, "volumeStiffnessE")
            col.label(text="Damping")
            col.prop(props, "volumeDamping")
            col.prop(props, "volumeDampingE")
            
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
            col.prop_search(props, "pinGroup", bpy.context.object, "vertex_groups")
            """
            row = layout.row()
            row.label(text="Cache:", icon='DISK_DRIVE') 
            col = layout.grid_flow(row_major=True, columns=1, even_columns=True, even_rows=True, align=False)
            #col.alignment = 'CENTER'
            col.prop(props, "cache")
            col.operator("object.bake_button")
            col.operator("object.bake_from_current_frame_button")
            col.operator("object.delete_all_bakes_button")
            """
    class BakeButton(bpy.types.Operator):
        bl_idname = "object.bake_button"
        bl_label = "Bake"

        def execute(self, context):
            return {'FINISHED'}
        
    class BakeFromCurrentFrameButton(bpy.types.Operator):
        bl_idname = "object.bake_from_current_frame_button"
        bl_label = "Bake from current frame"

        def execute(self, context):
            return {'FINISHED'}
        
    class DeleteAllBakesButton(bpy.types.Operator):
        bl_idname = "object.delete_all_bakes_button"
        bl_label = "Delete all bakes"

        def execute(self, context):
            return {'FINISHED'}
        
    bpy.utils.register_class(TetrahedralWorkshopButton)
    bpy.utils.register_class(TetrahedralWorkshopPanel)
    bpy.utils.register_class(BakeButton)
    bpy.utils.register_class(BakeFromCurrentFrameButton)
    bpy.utils.register_class(DeleteAllBakesButton)
    bpy.utils.register_class(TetProperties)
    bpy.types.Object.tet_properties = bpy.props.PointerProperty(type=TetProperties)

#//////////////////////////////////////////////////////////////////////////////////

def register():
    # Clear handlers when reloading script...
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.animation_playback_pre.clear()
    bpy.app.handlers.animation_playback_post.clear()
    bpy.app.handlers.render_pre.clear()
    
    bpy.utils.register_class(TetrahedralWorkshop)
    bpy.app.handlers.animation_playback_pre.append(on_playback_start)
    bpy.app.handlers.animation_playback_post.append(on_playback_stop)
    bpy.app.handlers.frame_change_pre.append(reset_positions)
    bpy.app.handlers.render_pre.append(on_render_pre)

def unregister():
    bpy.utils.unregister_class(TetrahedralWorkshop)
    
    # Should only remove the specific functions instead of clear...
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.animation_playback_pre.clear()
    bpy.app.handlers.animation_playback_post.clear()
    bpy.app.handlers.render_pre.clear()

if __name__ == "__main__":
    register()