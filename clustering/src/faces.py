"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *
from collections import defaultdict

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets

    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """

    n,d = X.shape

    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])

    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """

    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.

    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed

    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)

    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]

    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))

    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.

    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2c: implement (hint: use np.random.choice)
    return np.random.choice(points, k, replace=False)
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!

    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.

    Parameters
    --------------------
        points         -- list of Points, dataset

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2f: implement
    label_to_points = {}
    for p in points:
        if p.label in label_to_points:
            label_to_points[p.label].append(p)
        else:
            label_to_points[p.label] = [p]
    return [Cluster(v).medoid() for _, v in label_to_points.items()]
    ### ========== TODO : END ========== ###


def compute_assignments_kMeans(clusters, points):
    """Computes assignments of points to clusters based on min dist.
    Params:
    clusters: list of points that represent clusters
    points: points to assign
    Returns:
    dict, where k: v = point: assigned cluster index
    """
    assigns = {}
    for p in points:
        min_dist, idx = np.Inf, -1
        for i in range(len(clusters)):
            dst = p.distance(clusters[i])
            if dst < min_dist:
                min_dist, idx = dst, i
        if idx in assigns:
            assigns[idx].append(p)
        else:
            assigns[idx] = [p]
    return assigns


def kMeans(points, k, init='random', plot=False, verbose = False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.

    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable:
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm

    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """

    ### ========== TODO : START ========== ###
    # part 2c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).
    if init == 'random':
        cur_centroids = random_init(points, k)
        if verbose: print "random init in use"
    else:
        cur_centroids =  cheat_init(points)
        if verbose: print "cheat init in use"
    iters = 0
    prev_clusts = None
    while True:
        cur_cluster_set = ClusterSet([Cluster(v) for _, v
                                       in compute_assignments_kMeans(
                                           cur_centroids, points).items()])
        iters+=1
        if verbose: print "iters: {}".format(iters)
        if plot: plot_clusters(cur_cluster_set, 'Plot of kMeans Clusters, iteration {} using {} init'.format(iters, init),
                               ClusterSet.centroids)
        if prev_clusts is not None and cur_cluster_set.equivalent(prev_clusts):
            if verbose: print "Done in {} iters".format(iters)
            return cur_cluster_set
        else:
            prev_clusts = cur_cluster_set
            #iters+=1

            cur_centroids = cur_cluster_set.centroids()




    k_clusters = ClusterSet()
    return k_clusters
    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False, verbose = False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 2e: implement
    if init == 'random':
        cur_centroids = random_init(points, k)
        if verbose: print "random init in use"
    else:
        cur_centroids =  cheat_init(points)
        if verbose: print "cheat init in use"
    iters = 0
    prev_clusts = None
    while True:
        for c in cur_centroids:
            assert(c in points)
        # compute assignments
        cur_cluster_set = ClusterSet([Cluster(v) for _, v
                                      in compute_assignments_kMeans(
                                          cur_centroids, points).items()])
        iters+=1
        if plot: plot_clusters(cur_cluster_set, 'KMediods Clusters, iteration {} using {} init'.format(iters, init),
                               ClusterSet.medoids)
        if verbose: print "iters: {}".format(iters)
        if prev_clusts is not None and cur_cluster_set.equivalent(prev_clusts):
            if verbose: print "Done in {} iters".format(iters)
            return cur_cluster_set
        else:
            prev_clusts = cur_cluster_set
            #iters+=1
        # if plot: plot_clusters(cur_cluster_set, 'KMediods Clusters, iteration {} using {} init'.format(iters, init),
        #                        ClusterSet.medoids)
        cur_centroids = cur_cluster_set.medoids()

    k_clusters = ClusterSet()
    return k_clusters
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    ### ========== TODO : START ========== ###
    # part 1: explore LFW data set
    X, y = get_lfw_data()
    mean_face = np.mean(X, axis = 0)
    U, mu = PCA(X)
    assert(np.sum(np.abs(mean_face - mu)) == 0)
    show_image(vec_to_image(mu)) #PART A
    num_eigenfaces_to_plot = 12
    plot_gallery([vec_to_image(U[:,i])
                  for i in xrange(num_eigenfaces_to_plot)]) #PART B
    for l in [1,10,50,100,500,1288]:
        Z, Ul = apply_PCA_from_Eig(X, U, l, mu)
        X_rec = reconstruct_from_PCA(Z, Ul, mu)
        plot_gallery([vec_to_image(X_rec[i])
                      for i in xrange(num_eigenfaces_to_plot)]) #PART C
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part 2d-2f: cluster toy dataset
    print "generating data for clustering"
    np.random.seed(1234)
    pts = generate_points_2d(20)
    cluster_set = kMeans(pts, 3, plot = False, verbose = False) # 2
    print "kmeans rand init score: {}".format(cluster_set.score())
    another_cluster_set = kMedoids(pts, 3, plot = False, verbose = False) #2
    print "k medoids rand init score: {}".format(another_cluster_set.score())
    km_clust_2 = kMeans(pts, 3, init = 'cheat', plot = False, verbose = False) #2
    print "k means cheat init score: {}".format(km_clust_2.score())
    k_med_clust_2 = kMedoids(pts, 3, init='cheat', plot = False, verbose = False) #2
    print "k medoids cheat init score: {}".format(k_med_clust_2.score())

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part 3a: cluster faces
    np.random.seed(1234)
    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)
    kmeans_scores, kmed_scores = [], []
    kmeans_times, kmed_times = [], []
    import time
    for i in range(10):
        print "running k-means and k-medoids for the {}th time".format(i+1)
        t = time.time()
        cluster_set = kMeans(points, 4)
        kmeans_times.append(time.time()-t)
        kmeans_scores.append(cluster_set.score())
        t = time.time()
        kmed_set = kMedoids(points, 4)
        kmed_times.append(time.time()-t)
        kmed_scores.append(kmed_set.score())
    means_avg, means_max, means_min = np.mean(np.array(kmeans_scores)), max(kmeans_scores), min(kmeans_scores)
    med_avg, med_max, med_min = np.mean(np.array(kmed_scores)), max(kmed_scores), min(kmed_scores)
    kmeans_time = np.mean(np.array(kmeans_times))
    kmed_time = np.mean(np.array(kmed_times))
    print "kmeans time: {}".format(kmeans_time)
    print "kmed time: {}".format(kmed_time)
    print "K means average: {}, max: {}, min: {}".format(means_avg,
                                                         means_max, means_min)
    print "K medoids average: {}, max: {}, min: {}".format(med_avg,
                                                           med_max, med_min)

    # part 3b: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)
    X2, y2 = util.limit_pics(X, y, [4, 13], 40)
    l_kmeans = {}
    l_kmed = {}
    for l in range(1,42):
        if l % 5 == 0: print "iteration: l = {}".format(l)
        Z, Ul = apply_PCA_from_Eig(X2, U, l, mu)
        X2_rec = reconstruct_from_PCA(Z, Ul, mu)
        points = build_face_image_points(X2_rec, y2)
        kmeans_clust = kMeans(points, 2, init='cheat')
        kmed_clust = kMedoids(points, 2, init='cheat')
        l_kmeans[l] = kmeans_clust.score()
        l_kmed[l] = kmed_clust.score()
    plt.plot(list(l_kmeans.keys()), list(l_kmeans.values()), 'r', label='K means')
    plt.plot(list(l_kmed.keys()), list(l_kmed.values()), 'b', label='K medoids')
    plt.title('K-means and K-medoids score with respect to principal components')
    plt.xlabel('Number of principal components')
    plt.ylabel('Clustering score')
    plt.legend()
    plt.show()
    print l_kmed.items()

    # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
    max_score, min_score = (-1, None, None), (np.Inf, None, None)
    min_tup, max_tup = (None, None, []), (None, None, [])
    np.random.seed(1234)
    for i in range(0,19):
        for j in range(0,19):
            if i != j:
                if i % 5 == 0 and j % 5 == 0:
                    print "considering groups {} and {}".format(i,j)
                X_ij, y_ij = util.limit_pics(X, y, [i,j], 40)
                points = build_face_image_points(X_ij, y_ij)
                med_clust = kMedoids(points, 2, init='cheat')
                score = med_clust.score()
                if score < min_score[0]:
                    min_score = (score, i, j)
                if score > max_score[0]:
                    max_score = (score, i, j)
    print max_score
    print min_score
    assert(min_score[1] == 4 and min_score[2] == 5)
    plot_representative_images(X, y, [min_score[1], min_score[2]],
                               title = 'min score images')
    assert(max_score[1] == 9 and max_score[2] == 16)
    plot_representative_images(X, y, [max_score[1], max_score[2]],
                               title = 'max score images')





    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
