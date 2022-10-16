def count_leaves(dt, c=[0,0]):
    """
	Count number of non-leaf and leaf branches.
	
	Parameter:
	dt -- the decision tree
	c -- a counter
	
	Return:
	c -- a count for both non-leeaves and leaves
	"""
    c[0] += 1
    leaves = dt.children()
    for leaf in leaves: 
        branches = dt[leaf].values()
        for branch in branches: 
            if isinstance(branch, dict):
                count_leaves(branch, c)
            else:
                c[1] += 1
    return c

def print_statistics(dt, t, tr, te, trs, tes):
    """
    Prints diagnostics regarding decision tree.
    
    Parameter:
    dt -- the decision tree
    t -- the time it took to generate dt
    tr -- classification ability of training data
    te -- classification ability of novel (test) data
    trs -- number of training examples
    tes -- number of testing examples
    """
    #s, d = count_leaves(dt) # splits and decisions
    print(f"Using {trs} training examples and {tes} testing examples.")
    #print(f"Tree contains {s} non-leaf nodes and {d} leaf nodes.")
    print("Took {:.2f} seconds to generate.".format(t))
    print(f"Was able to classify {tr}% of training data.")
    print(f"Was able to classify {te}% of testing data.\n")