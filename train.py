# -*- coding: utf-8 -*-

import caffe

solver_path = 'examples/SRCNN/SRCNN_solver.prototxt'
solver = caffe.SGDSolver(solver_path);
solver.solve();
