import os

from absl import flags, app, logging
from absl.flags import FLAGS


flags.DEFINE_string('dataset','','dataset path')
flags.DEFINE_string('output_file','classes.txt','output file name')

datasets = ['./bounding_box_train',
            './bounding_box_test']
def main(_argv):
    file_class = []
    for i in datasets:
        if os.path.exists(i):
            file_class.extend([int(j.split("_")[0]) for j in os.listdir(i)])
    classes = list(set(file_class)) 
    with open(FLAGS.output_file, 'w') as fp:
        for class_name in classes[:-2]:
            fp.write(str(class_name)+"\n")
if __name__ == '__main__':
    try:
        app.run(main)
    except:
        pass 