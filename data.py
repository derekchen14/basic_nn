import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


def generate(seed, moons):
  np.random.seed(seed)
  X, y = sklearn.datasets.make_moons(moons, noise=0.20)
  return X, y

def gimme():
  f = np.array([[0.74346118, 0.46465633],
 [ 1.65755662, -0.63203157],
 [-0.15878875,  0.25584465],
 [-1.088752  , -0.39694315],
 [ 1.768052  , -0.25443213],
 [ 1.95416454, -0.12850579],
 [ 0.93694537,  0.36597075],
 [ 0.88446589, -0.47595401],
 [ 0.80950246,  0.3505231 ],
 [ 1.2278091 , -0.64785108],
 [-0.38454276,  0.50916381],
 [ 0.09252135, -0.31618454],
 [ 1.79531658, -0.32235591],
 [ 1.43861749, -0.15796611],
 [-0.82364866,  0.86822754],
 [ 0.99633397,  0.1731019 ],
 [ 0.66388701,  0.94659669],
 [ 0.13229471, -0.26032619],
 [ 0.2482245 ,  0.7860477 ],
 [-1.00392102,  1.15207238],
 [ 2.08208438,  0.00715606],
 [ 0.87081342, -0.4366643 ],
 [ 0.37268327,  1.01743002],
 [ 1.26735927, -0.11813675],
 [-0.13270154,  1.26653562],
 [ 0.20331   ,  0.19519454],
 [ 1.98373996, -0.11222315],
 [ 1.82749513, -0.03085446],
 [-0.03857867,  0.0838378 ],
 [ 0.03351023,  0.63113817],
 [ 0.94193283,  0.63204507],
 [-0.39131894,  0.40925201],
 [ 0.88357043, -0.35868845],
 [-0.01141219,  0.30437635],
 [ 0.75877114,  0.76057045],
 [ 1.79414416,  0.28323389],
 [ 0.56116634, -0.0330033 ],
 [ 0.87161309,  0.01715969],
 [-0.75191922,  0.63798317],
 [-0.21911253,  0.49662864],
 [ 0.63711933, -0.55537183],
 [-0.25531442,  0.83953933],
 [ 0.57753017,  0.64564015],
 [ 0.15931878, -0.02835184],
 [ 1.53296943, -0.36277826],
 [-0.24648981,  1.09136047],
 [ 1.16443301,  0.01495781],
 [-0.70574528,  0.54883003],
 [ 0.16919147, -0.30895665],
 [ 1.0717818 , -0.40141988],
 [-0.8970433 ,  0.87690996],
 [ 0.4828491 , -0.21452374],
 [ 2.25536302,  0.02862685],
 [-0.62523133,  0.03868576],
 [ 1.22821377, -0.50119159],
 [ 0.84248307,  0.55728315],
 [ 0.45857236,  0.5017019 ],
 [ 0.98031957, -0.56811367],
 [ 0.1059936 ,  0.90514125],
 [-0.21582418,  1.03521642],
 [ 0.06721632, -0.1649077 ],
 [-1.07873435,  0.36644163],
 [ 1.60172165, -0.37604995],
 [ 1.02592325,  0.42143427],
 [ 1.06739115, -0.38783511],
 [-1.35462041,  0.28524762],
 [-0.20784982,  1.09043495],
 [ 1.61652485, -0.29469483],
 [ 0.26375409,  0.91508367],
 [-0.99805184,  0.62420544],
 [ 0.62273618, -0.52804644],
 [-1.0873102 ,  0.78128608],
 [ 0.01262924, -0.59715374],
 [-0.52953439,  0.69307316],
 [ 0.78362442, -0.25844144],
 [-0.94262451,  0.57258351],
 [ 0.09048712,  0.0890939 ],
 [ 0.99716574,  0.35017425],
 [ 0.4630177 ,  0.86392418],
 [ 0.71787709, -0.09708361],
 [ 2.13330659,  0.11200406],
 [-0.41467068,  0.92254691],
 [ 0.6233932 , -0.69422694],
 [ 2.04970274,  0.66368306],
 [-0.00353234,  0.21487064],
 [-0.27631969,  1.34161045],
 [ 0.82262609, -0.02317445],
 [-0.46610015,  0.98764879],
 [ 0.64426474, -0.36209808],
 [ 1.96682571,  0.2646737 ],
 [ 0.71060915,  0.80990546],
 [ 1.12820353,  0.4664342 ],
 [ 1.99150162,  0.02534858],
 [-0.66342048,  0.85301441],
 [ 2.0436285 ,  0.24563453],
 [ 1.77377397, -0.10513907],
 [ 1.773464  , -0.34102513],
 [ 0.66137686, -0.31314104],
 [-1.15442774,  0.40574243],
 [ 0.04167562, -0.07462092],
 [ 1.40426435, -0.93206382],
 [ 1.99317676,  0.48903983],
 [ 0.17673342,  1.3178874 ],
 [ 1.12344625, -0.09556327],
 [-0.64018301,  0.75214137],
 [ 0.17295579,  0.60135526],
 [-0.97644617,  0.03612864],
 [-0.56357758,  1.15774717],
 [ 1.60440089, -0.35116358],
 [ 0.13387667,  0.6944329 ],
 [-0.59909677,  0.76903039],
 [ 0.1023533 ,  1.09326207],
 [-0.22047436,  1.28343927],
 [-0.70416708,  0.30649274],
 [ 0.95709601,  0.30502143],
 [ 1.65936346, -0.70351567],
 [ 0.18911691,  0.64887424],
 [ 2.02773677,  0.25021451],
 [ 0.6515764 , -0.40677494],
 [ 0.55688998,  0.26120887],
 [ 0.81816111,  0.78952806],
 [-0.48367053,  0.43679813],
 [-0.14739828,  0.22556193],
 [ 0.11834786,  0.99156023],
 [-0.25253387,  0.18776697],
 [-0.93313522,  0.73385959],
 [ 0.6975216 , -0.11832611],
 [ 0.33332321,  0.14006592],
 [ 1.06519327, -0.38867949],
 [ 1.9369961 ,  0.63112161],
 [ 1.05840957,  0.51858443],
 [-0.50840939,  0.55259494],
 [-1.32109805,  0.51437657],
 [ 0.29449971, -0.26078938],
 [ 1.33653621, -0.18005761],
 [ 1.51241178,  0.11081331],
 [ 1.01934807, -0.17993629],
 [-1.13305483,  0.11109962],
 [ 2.07463826,  0.51253705],
 [ 0.73451679,  0.5346233 ],
 [-0.12213442,  0.15292037],
 [-0.0557186 ,  0.57286794],
 [ 0.45046033,  1.09585861],
 [-0.7204608 ,  1.01733354],
 [-0.33698825,  0.89060661],
 [ 1.0628775 ,  0.17231496],
 [ 0.34005355,  0.32486358],
 [ 1.24491552, -0.5137574 ],
 [ 0.30966003,  1.16677531],
 [-0.06114159, -0.02921072],
 [ 0.48281721, -0.43196099],
 [ 1.68734249, -0.6872367 ],
 [ 0.80862106,  0.28415372],
 [ 0.29809162,  0.82211432],
 [ 0.8496547 , -0.30507345],
 [-0.3802171 ,  0.88414623],
 [ 1.32734432, -0.48056888],
 [ 0.23337057,  0.10750568],
 [ 0.68841773,  1.15068264],
 [ 0.6779624 ,  0.78024482],
 [ 0.3395913 , -0.02223857],
 [ 1.30440877, -0.52950917],
 [ 0.75307594,  0.8526869 ],
 [ 1.4298847 , -0.21080222],
 [ 0.55631903, -0.70781481],
 [ 1.45384401,  0.12718529],
 [ 0.3203754 ,  0.87271389],
 [ 0.53148147, -0.27424077],
 [ 1.51658699, -0.45069719],
 [ 0.99826403, -0.80979075],
 [ 0.63918299,  0.96606739],
 [-1.2855903 ,  0.10677262],
 [-1.07840959,  0.56402523],
 [-0.57716798,  0.2942259 ],
 [ 0.25403599, -0.00644002],
 [ 0.91722632, -0.29657499],
 [ 1.43380709,  0.69183071],
 [-0.70851168,  0.49617855],
 [-0.64683386,  0.46971252],
 [ 0.30143461,  0.76398572],
 [ 1.48069489, -0.3572808 ],
 [-1.02663961,  0.41265823],
 [ 1.89660871,  0.25413209],
 [ 2.04251223, -0.46074593],
 [ 1.92673019,  0.40817963],
 [ 0.35766276,  1.0872811 ],
 [ 0.1240315 ,  0.67672995],
 [ 0.97332087, -0.70530678],
 [-0.72894228,  0.44179419],
 [-0.69863061,  0.77620293],
 [-0.93516752,  0.43520803],
 [ 0.45166927,  1.00185497],
 [ 0.87629641,  0.28951999],
 [ 0.88155818,  0.23925957],
 [-0.07795147,  0.27995261],
 [-0.56365899,  0.8918972 ],
 [ 1.6049806 ,  0.13835516],
 [ 0.27695668,  0.01210816],
 [ 0.25919429,  1.04104213],
 [ 1.5215205,  -0.1258923 ]])
  g = np.array([0,1,1,0,1,1,0,1,0,1,0,1,1,1,0,0,0,1,0,0,1,1,0,1,0,1,1,1,1,0,0,0,1,1,0,1,
   1,0,0,1,1,0,0,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,
   1,0,1,0,0,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,0,1,1,1,0,0,0,1,0,0,1,0,0,0,
   0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,
   1,1,0,0,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,0,1,1,1,0,0,1,
   0,0,0,0,0,0,1,0,1,1,0,1])
  return f, g

def plot_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    probs, a1 = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(probs, axis=1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.PuOr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PiYG)

def back_propagate_softmax(out, delta, a1):
  delta[range(num_examples), out] =- 1        # (200x2)
  d_weight = (a1.T).dot(delta)                # (3x2)
  d_bias = np.sum(delta, axis=0)              # (1x2)
  return d_weight, d_bias

def back_propagate_tangent(out, a2, a1):
  future_error = a2.dot(out.T)
  tanh_gradient = (1 - np.square(a1))
  delta = future_error * tanh_gradient        # (200x3)
  d_weight = (X_train.T).dot(delta)           # (2x3)
  d_bias = np.sum(delta, axis=0)              # (1x3)
  return d_weight, d_bias

def back_propagate_sigmoid(x, y, a2, a1):
  future_error = a2.dot(out.T)
  sigmoid_gradient = y(1 - y)
  delta = future_error * aigmoid_gradient     # (200x3)
  d_weight = (x.T).dot(delta)                # (2x3)
  d_bias = np.sum(delta, axis=0)              # (1x3)
  return d_weight, d_bias

def back_propagate_relu(x, y, a2, a1):
  future_error = a2.dot(out.T)
  relu_gradient = 0 if (y < 0) else 1
  delta = future_error * relu_gradient        # (200x3)
  d_weight = (x.T).dot(delta)                # (2x3)
  d_bias = np.sum(delta, axis=0)              # (1x3)
  return d_weight, d_bias

    # dW2, db2 = back_propagate_softmax(y_train, a2, a1)
    # dW1, db1 = back_propagate_tangent(model['W2'], a2, a1)

def early_stopping(model):
  loss_diff = 1
  counter = 0
  last_loss = 0

  while loss_diff > 1.0e-06:
    data_loss = calculate_loss(model)/num_examples
    loss_diff = np.abs(last_loss - data_loss)
    last_loss = data_loss
    print loss_diff

def post_time(start):
  finish = time.time()
  training_time = start - finish
  minutes, seconds = divmod(training_time, 60)
  hours, minutes = divmod(minutes, 60)
  print "%d hours, %02d minutes, %02d seconds" % (hours, minutes, seconds)
