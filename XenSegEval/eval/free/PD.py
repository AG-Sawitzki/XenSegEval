def getid(samp_path):
    return samp_path.split("/")[-1].split(".")[0]


def create_pmask(segpath, r):
    """converts a standard segmentation mask into one with a dilated area around
    the center of the original mask"""
    seg = imread(segpath)
    mask = np.zeros_like(seg)
    rps = regionprops(seg)

    # get center of each cell and set the value to one
    centroids = np.array(
        list(map(lambda x: np.array(x.centroid).astype(int), rps))
    )
    mask[centroids[:, 0], centroids[:, 1]] = 1

    # dilate around the center of the cell with the radius size and binarize the mask
    mask = binary_dilation(mask, disk(r))

    return mask


def compute_masks(sid, radii, methods, datapath):
    """gets all probability masks (all radii sizes X all segmentation methods) for a single sample
    return a dictionary containing a dictionary for each radii size, which containins the pmask
    for each segmentation method plus the average of all the masks across methods"""
    masks = {}
    for r in radii:  # for each radius size
        rmasks = {}

        for m in methods:  # for each segmentation method
            segpath = glob(os.path.join(datapath, m, sid + "*"))[0]
            pmask = create_pmask(segpath, r)  # create pmask
            rmasks[m] = pmask

        # get average pmask of all methods for a single radius
        stack = np.stack(list(rmasks.values()))
        avg_mask = stack.mean(0)
        rmasks["mean"] = avg_mask

        # save masks from each method with radius
        masks[r] = rmasks

    return masks


def write_pmasks(sample_ids, radii, methods, datapath, save_dir):
    """main function to create and save probability masks"""
    print(f"Writing probability masks for {len(sample_ids)} samples")

    for sid in tqdm(sample_ids):
        # save path for sample
        save_path = os.path.join(save_dir, f"{sid}.pkl")
        # dont recreate files
        if os.path.exists(save_path):
            continue

        # get masks for sample
        sid_masks = compute_masks(sid, radii, methods, datapath)
        with open(save_path, "wb") as handle:
            pickle.dump(sid_masks, handle)

    print(f"Proability masks saved to {save_dir}")


def filter_mask(mask, avg_labs):
    """filter probability mask for a single method (mask) using the thresholded, averaged, proability mask"""
    # copy
    filtered = copy.deepcopy(mask)
    mask_lab = label(mask)
    rps = regionprops(mask_lab)

    for rp in rps:  # for each cell region
        # get cell center coordinates from single-method mask
        coords = rp.coords
        # use those coords to get values of averaged mask
        vals = avg_labs[coords[:, 0], coords[:, 1]]
        uniq, counts = np.unique(vals, return_counts=True)

        # ignore background
        if uniq[0] == 0:
            uniq = uniq[1:]
            counts = counts[1:]

        n_unique = len(uniq)

        # if more than 1 value, zero out pixels in mask that do not equal the most common value
        if n_unique > 1:
            amax = np.argmax(counts)
            top_val = uniq[amax]
            idxs = np.where(vals != top_val)
            to_zero = coords[idxs, :][0]
            filtered[to_zero[:, 0], to_zero[:, 1]] = False

    return filtered


def filter_pmasks(sample_ids, pmask_save_dir, filtered_save_dir, min_num_agree, methods):
    print(f"Filtering probability masks for {len(sample_ids)} samples")

    for sid in tqdm(sample_ids):  # for each sample
        # open pickle file containing probability masks
        with open(os.path.join(pmask_save_dir, f"{sid}.pkl"), "rb") as handle:
            data = pickle.load(handle)

        filtered_masks = {}
        # filter averaged probability masks
        for r, masks in data.items():  # for each radius, set of probability masks
            avg = masks["mean"]
            # threshold average mask to only keep pixels labeled by at least min_num_agree seg. methods
            avg_threshd = (avg >= (min_num_agree / len(methods)))
            avg_labs = label(avg_threshd)

            r_filtered_masks = {}
            for m in methods:  # for each probability method
                # use thresholded, averaged probability mask
                r_filtered_masks[m] = filter_mask(masks[m], avg_labs)

            # get new average mask
            new_stack = np.stack(list(r_filtered_masks.values()))
            new_avg = new_stack.mean(0)
            r_filtered_masks["mean"] = new_avg
            filtered_masks[r] = r_filtered_masks

        # save new probability masks
        if os.path.exists(os.path.join(filtered_save_dir, f"{sid}.pkl")):
            continue
        with open(os.path.join(filtered_save_dir, f"{sid}.pkl"), "wb") as handle:
            pickle.dump(filtered_masks, handle)

    print(f"Filtered probability masks saved to {filtered_save_dir}")


def eval_mask(gt, m):
    """returns precision and recall for a pair of masks, see diagram above"""
    rps = regionprops(m)
    coords = list(map(lambda x: x.coords, rps))
    correct = 0

    for c in coords:
        correct += (gt[c[:, 0], c[:, 1]]).max()
    precision = correct / len(rps)

    gt_labs = label(gt)
    gt_rps = regionprops(gt_labs)
    coords = list(map(lambda x: x.coords, gt_rps))
    correct = 0

    for c in coords:
        correct += (m[c[:, 0], c[:, 1]]).max() > 0
    recall = correct / len(gt_rps)

    assert precision <= 1
    assert precision >= 0
    assert recall <= 1
    assert recall >= 0

    return precision, recall


def evaluate_masks(sample_ids, filtered_pmask_save_dir, radii, min_num_agree, num_methods):
    precision = {}
    recall = {}
    print(f"Computing precision and recall for {len(sample_ids)} samples")
    for sid in tqdm(sample_ids):
        data_load_path = os.path.join(filtered_pmask_save_dir, f"{sid}.pkl")
        with open(data_load_path, "rb") as handle:
            data = pickle.load(handle)

        sid_precisions = {r: {} for r in radii}
        sid_recalls = {r: {} for r in radii}

        for r, masks in data.items():

            avg = masks["mean"]
            avg_thresh = (avg >= (min_num_agree / num_methods))

            for name, mask in masks.items():
                if name == "mean":
                    continue
                labd_mask = label(mask)
                prec, rec = eval_mask(avg_thresh, labd_mask)
                sid_precisions[r][name] = prec
                sid_recalls[r][name] = rec

        precision[sid] = sid_precisions
        recall[sid] = sid_recalls

    return precision, recall


def plot_precision_recall(precision, recall, methods, sample_ids):
    points = {m: [] for m in methods}

    for sid in sample_ids:
        for i, rad in enumerate(radii):
            for m in methods:
                p = precision[sid][rad][m]
                r = recall[sid][rad][m]
                points[m].append((r, p))

    fig, ax = plt.subplots(1, len(methods), figsize=(20, 2))
    fig.text(0.5, -0.1, 'precision', ha='center', va='center')
    fig.text(0.1, 0.5, 'recall', ha='center', va='center', rotation='vertical')
    for method, a in zip(methods, fig.axes):
        a.set_xlim(0.3, 1)
        a.set_ylim(0.5, 1)
        a.set_title(method)

        xs = [x for x, y in points[method]]
        ys = [y for x, y in points[method]]

        sns.kdeplot(x=xs, y=ys, clip=(0, 1), ax=a)
        a.scatter(xs, ys)
    plt.show()
