import sys
import json

class Node:
    def __init__(self,id,x,y,z):
        self.id=id
        self.x=x
        self.y=y
        self.z=z

    def __str__(self):
        jobj={'node_id':self.id,
              'x':self.x,
              'y':self.y,
              'z':self.z}
        return json.dumps(jobj)

class Element:

    id2nodes=None

    def __init__(self,id,size):
        self.id=id
        self.node_ids=[-1]*size
        self.size=size

    def check_legal(self):
        if len(self.node_ids) != self.size:
            return False
        for ids in self.node_ids:
            if ids<0:
                return False
        return True

    def __str__(self):
        jobj={'element_id':self.id,
              'nodes':[]}
        for i in range(self.size):
            jobj['nodes'].append(json.loads(str(id2nodes[self.node_ids[i]])))

        return json.dumps(jobj)




with open(sys.argv[1]) as f:
    jobj=json.load(f)
    for key in jobj:
        #print("{} has {} lines".format(key,len(jobj[key])))
        print("##############################{} has {} lines##############################"
              .format(key,len(jobj[key])))
        min_l=min(len(jobj[key]),5)

        if min_l == 1:
            print(jobj[key])
        else:
            for i in range(min_l):
                print(jobj[key][i])



    push_id2triplets={}
    for i,element in enumerate(jobj['push_elements']):
        if not element['element_id'] in push_id2triplets:
            push_id2triplets[element['element_id']]=[]
        push_id2triplets[element['element_id']].append(element)


    count=0
    for id in push_id2triplets:
        print(push_id2triplets[id])
        count+=1
        if count>=5:
            break

    #indexing node
    id2nodes={}
    for item in jobj['nodes']:
        id2nodes[item['node_id']]=Node(id=item['node_id'],
                                       x=item['x'],
                                       y=item['y'],
                                       z=item['z'])
    print('####Nodes: id##########')
    node_ids=sorted(id2nodes.items(),key=lambda x:x[1].id)
    for i in range(5):
        print(node_ids[i][1])


    print('####Nodes: x##########')
    node_ids=sorted(id2nodes.items(),key=lambda x:x[1].x)
    for i in range(5):
        print(node_ids[i][1])

    print('####Nodes: y##########')
    node_ids=sorted(id2nodes.items(),key=lambda x:x[1].y)
    for i in range(5):
        print(node_ids[i][1])

    print('####Nodes: z##########')
    node_ids=sorted(id2nodes.items(),key=lambda x:x[1].z)
    for i in range(5):
        print(node_ids[i][1])


    #indexing_element
    push_elements={}
    for i,item in enumerate(jobj['push_elements']):
        if not item['element_id'] in push_elements:
            push_elements[item['element_id']]=Element(id=item['element_id'],size=3)
        push_elements[item['element_id']].node_ids[int(item['idx'])-1]=item['node_id']

    push_elements=list(sorted(push_elements.items(),key=lambda x:x[1].id))
    push_elements[0][1].id2nodes=id2nodes

    print('#####Elemetnds: #####')
    for i in range(5):
        print(push_elements[i][1])

    #indexing_element
    surf_elements={}
    for i,item in enumerate(jobj['surf_elements']):
        if not item['element_id'] in surf_elements:
            surf_elements[item['element_id']]=Element(id=item['element_id'],size=6)
        surf_elements[item['element_id']].node_ids[int(item['idx'])-1]=item['node_id']

    surf_elements=list(sorted(surf_elements.items(),key=lambda x:x[1].id))
    surf_elements[0][1].id2nodes=id2nodes

    print('#####surf_elements: #####')
    for i in range(5):
        print(surf_elements[i][1])





