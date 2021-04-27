import numpy as np
partition = '/projects/f_mc1689_1/'

def get_network_info(parc,subcortical_split=True):
    if parc == 'glasser':
        network_def = np.loadtxt(partition + 'AnalysisTools/ColeAnticevicNetPartition/cortex_subcortex_parcel_network_assignments.txt')
        network_def = network_def[0:360]
        network_list = ['vis1','vis2','smn','con','dan','lan','fpn','aud','dmn','pmulti','vmulti','orbaff']
        network_mappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8,
                                'lan':6, 'dan':5, 'con':4, 'dmn':9, 'pmulti':10,
                                'vmulti':11, 'orbaff':12}
    elif parc == 'cabn':
        #print('Parcellation=cabnp w/718 regions')
        network_def = np.loadtxt(partition + 'AnalysisTools/ColeAnticevicNetPartition/cortex_subcortex_parcel_network_assignments.txt')
        # ljh: I like to keep subcortical and cortical seperate
        if subcortical_split:
            network_def[360::] = network_def[360::]+12
            network_list = ['vis1','vis2','smn','con','dan','lan','fpn','aud','dmn','pmulti','vmulti','orbaff',
                            'sc-vis1','sc-vis2','sc-smn','sc-con','sc-dan','sc-lan','sc-fpn','sc-aud','sc-dmn','sc-pmulti','sc-vmulti','sc-orbaff']
            network_mappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8,
                                'lan':6, 'dan':5, 'con':4, 'dmn':9, 'pmulti':10,
                                'vmulti':11, 'orbaff':12,
                                'sc-fpn':19, 'sc-vis1':13, 'sc-vis2':14, 'sc-smn':15, 'sc-aud':20,
                                'sc-lan':18, 'sc-dan':17, 'sc-con':16, 'sc-dmn':21, 'sc-pmulti':22, 
                                'sc-vmulti':23, 'sc-orbaff':24}
        else:
            network_list = ['vis1','vis2','smn','con','dan','lan','fpn','aud','dmn','pmulti','vmulti','orbaff']
            network_mappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8,
                                'lan':6, 'dan':5, 'con':4, 'dmn':9, 'pmulti':10,
                                'vmulti':11, 'orbaff':12}

    network_order = np.asarray(sorted(range(len(network_def)), key=lambda k: network_def[k]))
    network_order.shape = (len(network_order),1)
    networks = network_mappings.keys()
    reordered_network_affil = network_def[network_order]

        # Create a categorical palette to identify the networks
    network_palette = ['royalblue','slateblue','paleturquoise','darkorchid','limegreen',
                          'lightseagreen','yellow','orchid','r','peru','orange','olivedrab',
                      'royalblue','slateblue','paleturquoise','darkorchid','limegreen',
                          'lightseagreen','yellow','orchid','r','peru','orange','olivedrab']
    network_labels = []
    for i in range(len(reordered_network_affil)):
        idx = np.int(reordered_network_affil[i]-1)
        network_labels.append(network_palette[idx])
    return network_order,network_labels,network_def,network_list