from tweets_data import *
from constants import *
from textual_features import construct_textual_features
from final_dataset_construction import truncate_dataset, save_dataset, prepare_bert_style_dataset, prepare_string_matching_dataset
import argparse
from label_construction import LABELING_STRATEGIES, construct_labels
from dispatch_data import TARGET_SCOPES
from dispatch_center_features import add_dispatch_center_features, add_calendar_features, get_dispatch_dates


pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)


def parse_args():

    parser = argparse.ArgumentParser()

    # random seed
    parser.add_argument('--random_seed', action='store', type=int, default=42, 
                        help='random seed used to conduct these experiments.')

    # debugging subsample
    parser.add_argument('--n_sample_tweets', action='store', type=int, default=None,
                        help='If set, randomly subsample this many tweets before dataset construction. '
                        'Useful for quick debugging runs; leave unset to use the full dataset.')
    

    # general dispatch center consideration
    parser.add_argument('--general_dispatch_center_focus', type=str, default='patients',
                        choices=['calls', 'patients'],
                        help='Target unit: the dispatch-centre quantity counted per day and used for the '
                        'history features. Can be calls (one row per call in appels.csv, default) or '
                        'patients (one row per case in M_final.csv)')

    parser.add_argument('--target_scope', type=str, default='respiratory', choices=list(TARGET_SCOPES),
                        help='Case selection for the target. total: every call/patient. '
                        'respiratory: only cases whose main problem code (p.PP) is a respiratory or '
                        'influenza-like code (SHORT_LIST: 1008, 1502). p.PP exists only in the case '
                        'export, so a respiratory target is always counted there.')

    # truncation date
    parser.add_argument('--truncation_date', type=str, default='2023-01-01', 
                        help='End date for truncating the Twitter dataset (exclusive cutoff, format: YYYY-MM-DD).')

    # if add dispatch center feature or not
    parser.add_argument('--add_dispatch_center_feature', action=argparse.BooleanOptionalAction, default=True, help='If set to true, we add feature concerning the historical emerency calls as features.')
    
    
    ##############################
    # Dataset categories
    ##############################
    parser.add_argument('--dataset_style', action='store', type=str, default='matching',
                        help='Define the dataset that will be constructed. Potential values are: matching (keywords extracted using fuzzy matching approach) and BERT (just a concatenation of daily text)')

    parser.add_argument('--bert_embedding_model', action='store', type=str, default='Twitter/twhin-bert-base',
                        help='Location of the BERT model used to embed tweet if --dataset_style is set to BERT.')
    
    parser.add_argument('--pooling_strategy', action='store', type=str, default='mean',
                        choices=['mean', 'max', 'engagement'],
                        help='Strategy used to pool daily tweets. For BERT embeddings: how the per-tweet '
                        'vectors are pooled into the daily vector (mean: unweighted, default; '
                        'engagement: weighted by 1+log1p(engagement); max). For the lexicon approaches '
                        'with --aggregation rate: the weighting of the daily mean (mean or engagement). '
                        'Ignored with --aggregation count, which never weights.')

    parser.add_argument('--aggregation', action='store', type=str, default='count',
                        choices=['count', 'rate'],
                        help='Daily aggregation of the lexicon features when --dataset_style is matching. '
                        'count (default): daily counts — n_tweets, count_<entity> (tweets with >= 1 match), '
                        'count_any. rate: weighted mean match count per tweet (score_<entity>), kept for '
                        'the sensitivity analysis.')    

    parser.add_argument('--considered_language', action='store', default='all',
                        help='Define the language langugae used for BERT embeddings. Potential values: all, german, english, french.' )


    ########################
    ## Labeling parameters
    #########################
    parser.add_argument('--labeling_strategy', action='store', type=str, default="surge_detection",
                        choices=list(LABELING_STRATEGIES),
                        help='Define the target labels for the prediction task. Potential values include: '
                        'weekly_trend: binary label, 1 if the mean target volume of the 7 days after day t '
                        'exceeds the mean of the 7 days ending on day t. '
                        'weekly_trends: same comparison, but the future week moves with --horizon. '
                        'weekly_calls: number of patients/calls for the next 7 days.'
                        'surge_detection: binary labels indicating if next day will be a surge. A surge is defined as next week s mean patients/call volume exceeding'
                        'the rolling 4-week (28-day) mean by more than `surge_threshold The past window is always fixed at 4 weeks (28 days) for therolling baseline.')
    
    parser.add_argument('--surge_threshold', action='store', type=float, default=0.20)
    parser.add_argument('--horizon', action='store', type=int, default=3,
                        help='Forecast horizon: weeks ahead for the weekly_* labeling strategies, '
                        'days ahead for surge_detection.')

    
    ########################
    ## Saving parameters
    ########################
    parser.add_argument('--final_dataset_name', action='store', type=str, default='matching_horizon_3',
                        help='Name of the final dataset file to be saved.')

    args = parser.parse_args()

    return args



def main():

    # loading arguments
    args = parse_args()

    # construct raw dataset obtained after tweet rehydratation OR load the preprocessed dataset if it has already been constructed.
    # df = construct_tweets_data()
    df = pd.read_pickle(DATA_DIR / "tweets.pkl")

    # drop irrelevant metadata
    df = df[TWEET_ROWS_TO_KEEP]

    print(f"[main] Number of tweets loaded: {len(df)}.")

    # for debugging: optional reproducible subsample of the tweets
    if args.n_sample_tweets is not None:
        df = df.sample(n=args.n_sample_tweets, random_state=args.random_seed)
        print(f"[main] Subsampled {args.n_sample_tweets} tweets for debugging.")

    print(f"[main] Number of tweets used for dataset construction: {len(df)}.")

    # days present in the dispatch-centre data, used to restrict the daily
    # index to the intersection with the tweet dates (see _get_date_range)
    args.dispatch_dates = get_dispatch_dates(args.general_dispatch_center_focus)
    
    # transform data in the targeted dataset
    if args.dataset_style == 'matching':
        print('#'*90)
        print('## We start to construct the dataset using a fuzzy string matching approach')
        print('#'*90)
        df = prepare_string_matching_dataset(df, params=args)

    elif args.dataset_style == 'BERT':
        print('#'*90)
        print('## We start to construct the dataset for BERT-style model')
        print('#'*90)
        # only keep the date, the label and the textual features
        df = prepare_bert_style_dataset(df, params=args)

    # add dispatch center information
    if args.add_dispatch_center_feature:
        df = add_dispatch_center_features(df, params=args)
    else:
        print("[main] Dispatch-centre features disabled (--no-add_dispatch_center_feature).")

    # add specific time feature
    df = add_calendar_features(df)
        
    df = construct_labels(df, params=args)

    # truncate the dataset 
    df = truncate_dataset(df, params=args)

    print('There is a total of {}  days in the dataset'.format(len(df)))

    # save the final dataset
    save_dataset(df, args, output_dir=BASE_DIR / "FINAL_DATASET")

if __name__ == "__main__":
    main()



