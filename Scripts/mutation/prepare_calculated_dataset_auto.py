from attributes import Attributes
import os
import subprocess

BUG_FOLDERS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
]
BUG_FOLDERS_STR = ["f1f2f3f4f5f6f7f8f9f10f11f12f13f14f15"]
REPOSITORY = "PATH_TO/master_thesis_dataset"
LABELED_DATASET_PATH = "labeled_dataset"

CALCULATED_DATASET_TXT_PATH = "calculated_dataset_txt"
CALCULATED_DATASET_PATH = "cleared_dataset"
CKJM_EXT = "ext_2.2_j16_45"
CKJM_DIR = "PATH_TO_CKJM"
CKJM_JAVA_SDK = "PATH_TO/jdk1.6.0_45/bin/java"

CLASSES = "PATH_TO_SOURCES_OF_A_VERSION_OF_A_PROJECT/build/classes"
LIBS = "PATH_TO_SOURCES_OF_A_VERSION_OF_A_PROJECT/lib"
ORG_BRANCH = "A_VERSION_OF_A_PROJECT_org"
COMPILE_JAVA_VER = "java_x.x.x" # Be sure that the necessary java sdk activated in environment

BUG_STEPS = [0.5]
# BUG_STEPS = [0.4, 0.5]
# BUG_STEPS = [0.3, 0.4, 0.5]


wd = os.getcwd()

for bug_step in BUG_STEPS:
    for fdx, folders in enumerate(BUG_FOLDERS):
        os.chdir(REPOSITORY)
        subprocess.run(
            [
                "git",
                "checkout",
                "-b",
                ORG_BRANCH.split("org")[0]
                + "v_1_0_15_"
                + BUG_FOLDERS_STR[fdx]
                + "_"
                + str(bug_step)
                + "_"
                + COMPILE_JAVA_VER,
                "refs/remotes/origin/"
                + ORG_BRANCH.split("org")[0]
                + "v_1_0_15_"
                + BUG_FOLDERS_STR[fdx]
                + "_"
                + str(bug_step)
                + "_"
                + COMPILE_JAVA_VER,
            ]
        )
        labeled_dataset_path = os.path.join(
            LABELED_DATASET_PATH,
            ORG_BRANCH.split("org")[0]
            + "v_1_0_15_"
            + BUG_FOLDERS_STR[fdx]
            + "_"
            + str(bug_step)
            + ".csv",
        )
        calculated_dataset_txt_path = os.path.join(
            CALCULATED_DATASET_TXT_PATH,
            ORG_BRANCH.split("org")[0]
            + "v_1_0_15_"
            + BUG_FOLDERS_STR[fdx]
            + "_"
            + str(bug_step)
            + "_"
            + COMPILE_JAVA_VER
            + "_"
            + CKJM_EXT
            + ".txt",
        )
        calculated_dataset_path = os.path.join(
            CALCULATED_DATASET_PATH,
            ORG_BRANCH.split("org")[0]
            + "v_1_0_15_"
            + BUG_FOLDERS_STR[fdx]
            + "_"
            + str(bug_step)
            + "_"
            + COMPILE_JAVA_VER
            + "_"
            + CKJM_EXT
            + ".csv",
        )

        os.mkdir(LIBS)

        os.chdir(CLASSES)
        subprocess.run(["jar", "cf", "classes.jar", "."])
        subprocess.run(["mv", "classes.jar", LIBS])

        # os.chdir(os.path.join(LIBS, 'test'))
        # for p in os.listdir(os.path.join(LIBS, 'test')):
        #     subprocess.run(['mv', p, LIBS])

        os.chdir(REPOSITORY)
        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", "j"])

        os.chdir(CKJM_DIR)
        # subprocess.run(['find', CLASSES, '-name', "'*.class'", '-print', '|', CKJM_JAVA_SDK, '-Djava.ext.dirs=' + LIBS, '-jar', 'ckjm_ext_22.jar', '>>', calculated_dataset_txt_path], shell=True)
        f = open(calculated_dataset_txt_path, "w")
        proc1 = subprocess.run(
            ["find", CLASSES, "-name", "*.class", "-print"], stdout=subprocess.PIPE
        )
        proc2 = subprocess.run(
            [CKJM_JAVA_SDK, "-Djava.ext.dirs=" + LIBS, "-jar", "ckjm_ext_22.jar"],
            input=proc1.stdout,
            stdout=f,
        )
        f.close()

        os.chdir(REPOSITORY)
        subprocess.run(["git", "stash", "save"])
        subprocess.run(["git", "stash", "drop"])

        os.chdir(wd)
        calculated_dataset_txt = open(calculated_dataset_txt_path, "r")
        labeled_dataset = open(labeled_dataset_path, "r")

        labeled_dataset_dict = {}
        for line in labeled_dataset.readlines():
            att = line.split(";")
            labeled_dataset_dict[att[0]] = Attributes(line)
        labeled_dataset.close()

        calculated_dataset_txt_dict = {}
        att_name = ""
        cc_lines = []
        for line in calculated_dataset_txt.readlines() + ["org.end "]:
            att = line.split(" ")
            if (
                att[0].split(".")[0] == "com"
                or att[0].split(".")[0] == "org"
                or att[0].split(".")[0] == "javax"
                or att[0].split(".")[0] == "bsh"
                or att[0].split(".")[0] == "gnu"
                or att[0].split(".")[0] == "installer"
                or att[0].split(".")[0] == "net"
                or att[0].split(".")[0] == "util"
            ):
                if att_name != "":
                    try:
                        calculated_dataset_txt_dict[att_name] = Attributes(
                            cc_lines, labeled_dataset_dict[att_name].bugs
                        )
                    except IndexError as inst:
                        print(att_name)
                        print(type(inst))  # the exception instance
                        print(inst.args)  # arguments stored in .args
                        print(inst)
                    except KeyError:
                        pass
                att_name = att[0]
                cc_lines = []
            cc_lines.append(line)
        calculated_dataset_txt.close()

        calculated_dataset = open(calculated_dataset_path, "w")
        for attributes in calculated_dataset_txt_dict.keys():
            calculated_dataset.write(
                calculated_dataset_txt_dict[attributes].dataset_line()
            )
        calculated_dataset.close()
