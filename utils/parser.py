import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MixGCF")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="yelp2018",
                        help="Choose a dataset:[gowalla,amazon-book,amazon,yelp2018,ali]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== # 
    parser.add_argument("--gnn", nargs="?", default="ngcf",
                        help="Choose a recommender:[lightgcn, ngcf, simgcl, linkfnd]")
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')

    parser.add_argument('--sample_method', type=int, default=0, help='use sampled graphs, 0 for no sampling')
    parser.add_argument('--tau', type=float, default=0.15, help='Temperature')
    parser.add_argument('--lamb', type=float, default=0.2, help='Lambda of CL Loss')
    parser.add_argument('--eps', type=float, default=0.2, help='epsilon from noise vectors')
    parser.add_argument('--cll', type=int, default=1, help='number of CL layer')
    parser.add_argument('--fnk', type=int, default=2, help='top-k strategy for FN')
    parser.add_argument('--threshold', type=float, default=0.9, help='threshold strategy for FN')

    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")

    parser.add_argument("--ns", type=str, default='mixgcf', help="rns,mixgcf")
    parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")

    parser.add_argument("--n_negs", type=int, default=1, help="number of candidate negative")
    parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60]',
                        help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument("--context_hops", type=int, default=3, help="hop")

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument(
        "--out_dir", type=str, default="./weights/yelp2018/", help="output directory for model"
    )
    # ===== save embedding ===== #
    parser.add_argument('--emb_save', type=bool, default=True, help='save final embeddings')

    return parser.parse_args()
