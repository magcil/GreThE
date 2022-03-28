from deep_audio_features.bin import basic_test
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        nargs='+',
                        required=True,
                        help='Files to analyze')
    FLAGS = parser.parse_args()


    input = FLAGS.input
    arousal = "merged_3sec_arousal_transfered_fin.pt"
    preds, posts = basic_test.test_model(arousal, input[0], layers_dropped=0,
               test_segmentation=True, verbose=True)
    print(posts)
    