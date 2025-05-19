#!/usr/bin/env python

import pickle


def Material_dict():
    # Element name
    # d-band center, eV
    # d-band full width, eV
    # rd, A
    
    reference_data = {'Ag':{'d_cen':-4.1421, 'full_width': 4.5040, 'rd': 0.6606606606606606 },
                      'Au':{'d_cen':-3.6577, 'full_width': 5.8312, 'rd': 0.7607607607607607 },
                      #'Cd':{'d_cen':-8.6701, 'full_width': 3.4283 },
                      'Co':{'d_cen':-1.5905, 'full_width': 6.6969, 'rd': 0.5605605605605606 },
                      'Cr':{'d_cen':-0.0876, 'full_width': 7.2096, 'rd': 0.6306306306306306 },
                      'Cu':{'d_cen':-2.6521, 'full_width': 4.2446, 'rd': 0.4904904904904905 },
                      'Fe':{'d_cen':-0.9278, 'full_width': 7.1120, 'rd': 0.6206206206206206 },
                      #'Hf':{'d_cen': 2.0669, 'full_width':10.4322, },
                      'Ir':{'d_cen':-2.6636, 'full_width': 9.5022, 'rd': 0.8208208208208209 },
                      #'La':{'d_cen': 2.0866, 'full_width': 8.0907, },
                      'Mn':{'d_cen':-0.6036, 'full_width': 7.0846, 'rd': 0.5905905905905906 },
                      'Mo':{'d_cen':-0.0110, 'full_width': 9.0573, 'rd': 0.8608608608608609 },
                      'Nb':{'d_cen': 0.6878, 'full_width': 8.9762, 'rd': 0.9409409409409409 },
                      'Ni':{'d_cen':-1.6686, 'full_width': 5.3541, 'rd': 0.5205205205205206 },
                      'Os':{'d_cen':-1.9693, 'full_width':10.6399, 'rd': 0.8508508508508509 },
                      'Pd':{'d_cen':-2.0870, 'full_width': 5.6793, 'rd': 0.6706706706706707 },
                      'Pt':{'d_cen':-2.6369, 'full_width': 7.6381, 'rd': 0.7907907907907907 },
                      'Re':{'d_cen':-1.0633, 'full_width':10.9953, 'rd': 0.8808808808808809 },
                      'Rh':{'d_cen':-2.0874, 'full_width': 7.3926, 'rd': 0.7207207207207207 },
                      'Ru':{'d_cen':-1.6305, 'full_width': 8.0663, 'rd': 0.7507507507507507 },
                      'Sc':{'d_cen': 1.7032, 'full_width': 5.9997, 'rd': 0.9409409409409409 },
                      'Ta':{'d_cen': 1.1906, 'full_width':10.8913, 'rd': 1.021021021021021  },
                      'Ti':{'d_cen': 1.3243, 'full_width': 6.7421, 'rd': 0.7907907907907907 },
                      'V': {'d_cen': 0.5548, 'full_width': 6.9774, 'rd': 0.6906906906906907 },
                      'W': {'d_cen': 0.1732, 'full_width':10.7373, 'rd': 0.9409409409409409 },
                      'Y': {'d_cen': 2.2707, 'full_width': 7.8737, 'rd': 1.2512512512512513 },
                      #'Zn':{'d_cen':-7.3926, 'full_width': 2.4747, },
                      'Zr':{'d_cen': 1.6485, 'full_width': 8.9050, 'rd': 1.0710710710710711 }}
    
    with open('./d_center/tinnet/feature/MaterialDict.pkl', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    data = {k.decode('utf8'): v for k, v in data.items()}
    
    keys = reference_data.keys()
    
    properties = ['d_cen', 'full_width', 'rd']
    
    for key in keys:
        for prop in properties:
            data[key][prop] = reference_data[key][prop]
    
    return data
