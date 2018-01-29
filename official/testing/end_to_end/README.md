## End-to-End Testing of Official Models

**Note: This is so alpha, we had to wrap around and call it zeta. Consider yourself warned.**

In order to facilitate performance and convergence testing of the official Model Garden models, we provide a set of scripts and tools that can:

1. Spin up a Kubernetes cluster on Google Cloud Platform.
2. Deploy jobs to the cluster.
3. Run performance and convergence tests for Garden models.
4. Produce results for easy consumption or dashboard creation.

Want to run your own cluster and tests? Read on!

### Important notes
This is one of the most critical sections. Please read these carefully before you go any further:

* Throughout this guide, we will assume a working knowledge of [Kubernetes](https://kubernetes.io/) (k8s) and [Google Cloud Platform](https://cloud.google.com) (GCP). If you are new to either, please run through this [tutorial](https://kubernetes.io/docs/getting-started-guides/gce/) first. **Don't ignore this piece of advice;** if you have never used Kubernetes or set up a cluster, your life will be much simpler if you at least run through that tutorial and ensure that you have all the default configs and permissions set up.
* You will need access to a GCP account and project; if you don't have one, [sign up](https://console.cloud.google.com/start).
* If you intend to spin up GPU instances, make sure your account has the appropriate permissions and [quotas](https://pantheon.corp.google.com/iam-admin/quotas) to do so. You will get hard-to-interpret errors if you are requesting GPUs without sufficient quotas.


### Pre-requisites

You will need [gcloud](https://cloud.google.com/sdk/gcloud/) and [kubectl](https://kubernetes.io/docs/reference/kubectl/overview/) installed on your local machine in order to deploy and monitor jobs. Please follow the [Google Cloud SDK installation instructions](https://cloud.google.com/sdk/docs/) and the [kubectl installation instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl/).

We assume here that you are running Kubernetes v1.9, and that you have sufficient permissions on your GCP account to create and delete instances, register docker containers, and set up firewalls.

After installation, make sure to set gcloud to point to the GCP project you will be using:

```
gcloud config set project <project-id>
```

You also need to check out  this repository and branch onto the local machine that will be running all of the below commands:

```
git clone https://github.com/tensorflow/models.git
git checkout feature/kubeflow-testing
```

Paths below assume you are starting from the Official Models directory, so:

```
cd models/official
```


### Get a Kubernetes cluster

#### Option 1: Bring your own cluster

The tools in the workload directory create pods and jobs on an existing cluster-- which means you need access to a k8s cluster on GCP. The easiest way to do that is to use the [Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine/) (GKE), which has a handy UI. However, GKE does not (as of January 2018) support GPU nodes, so the GKE solution will only work if you are running tests on CPU nodes.

In any case, if you are using GKE and manually set up a cluster, or if you happen to already have access to a GCP k8s cluster previously created, you can skip the rest of this section on creating a k8s cluster, and instead just configure `gcloud` to access your cluster:

```
gcloud config set project <project-id>
gcloud config set region <region-id>
kubectl config use-context <cluster-id>
```

#### Option 2: Spin up a cluster

If you don't have a cluster already, you can use the scripts in the cluster directory to set one up:

1. In your local copy of the cluster directory, edit the text file cluster/cluster_config.sh to reflect your desired configuration details. (It may be the case that you don't need to change anything at all. If you installed kubectl in a non-standard location, update `KUBE_ROOT`; if you want to use a different region, update `KUBE_GCE_ZONE`, though note that GPUs are only [available in certain regions](https://cloud.google.com/compute/docs/gpus/); if you don't want to use a GPU cluster, update `USE_GPU`; and if you want a different number of worker nodes, update `NUM_NODES`. Note that you should only change the operating system variables if you know what you are doing, as we do not currently support other variants.)

2. All right, this is the wacky part, and I apologize that this is necessary ahead of time. In order to successfully spin up a cluster running Ubuntu, we had to modify the core k8s scripts. I have an [open question on Stack Overflow](https://stackoverflow.com/questions/48121852/kube-up-sh-fails-to-initialize-ubuntu-master-in-cluster-in-kubernetes-v1-9); please feel free to answer that or fix this if you know how.

For now, you need to add the following to `$KUBE_ROOT/cluster/gce/gci/configure.sh` (most likely for you, that is `$HOME/kubernetes/cluster/gce/gci/configure.sh`), next to all the other functions:

```
function special-ubuntu-setup {
 # Special installation required for ubuntu 16.04, as per Karmel's experience
 # on 2017-12-29
 apt-get install python-yaml

 # Install docker
 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
 add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
 apt-get update
 apt-cache policy docker-ce
 apt-get install -y docker-ce
}
```
Then, in "main loop" at the bottom of the same file, call that function before `download-kube-env`. The final main loop should look like this, with the added call to `special-ubuntu-setup` in the middle:

```
######### Main Function ##########
echo "Start to install kubernetes files"
set-broken-motd
KUBE_HOME="/home/kubernetes"
KUBE_BIN="${KUBE_HOME}/bin"

# This next line is added in by Karmel. download-kube-env is
# failing for lack of yaml, docker, etc.:
special-ubuntu-setup

download-kube-env
source "${KUBE_HOME}/kube-env"
if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
  download-kube-master-certs
fi
install-kube-binary-config
echo "Done for installing kubernetes files"
```

3. Now that the painful part is over, you should be able to spin up a cluster with the `setup_cluster.sh` script:

```
testing/end_to_end/cluster/setup_cluster.sh
```

You should see output from k8s indicating status. Note that aqcuiring and setting up GPUs can take several minutes. The final output should indicate that the cluster is up and running, with various API endpoints available; if it doesn't... debug.


### Start your jobs

Now that you have a cluster, we will create pods that run your training and testing jobs.

1. If you just went through the previous step of creating your cluster, kubectl will likely already be connected to it, but if for whatever reason that's not the case, make sure to point kubectl to your cluster:

```
kubectl config use-context <cluster-id>
```

(Not sure what your cluster ID is? Run `kubectl config view` and find the name of the cluster you just created, which is probably something like `project-name_tf-models-cluster-###`)

2. Create or locate the config directory for the model you are testing. If you are running an existing test, look in end_to_end/models for the model subdirectory of interest. If you are creating a new test, your best bet is to replicate what is done in an existing model test directory, and edit accordingly. For now, we will assume you are running the Resnet tests.

3. Navigate to the chosen test directory. This is not ideal, but important for now because of my limited understanding of Docker and how to work with directories when building Docker containers.

```
cd testing/end_to_end/models/resnet
```

4. Modify the appropriate YAML file to reflect your configuration preferences. Note that the `_cpu` YAML assumes you are running a CPU-only cluster, and `_gpu` assumes you have access to a CUDA- and nvidia-docker-enabled GPU cluster. Right now, these YAML files are mostly for show, as Resnet is not actually running multi-GPU. But you can see where you would change things when the age of multi-GPU is upon us.

5. Launch your jobs!

```
USE_GPU=true  # Or false, if you are using a CPU cluster
python testing/end_to_end/cluster/workload/launch_jobs.py --task_config_file=resnet_cifar10_$([ "$USE_GPU" == true ] && echo "gpu" || echo "cpu").yaml --results_dir=/tmp --docker_image_pattern="gcr.io/<you-project-id>/<your-docker-dir>/%s"
```
Note that the results directory is currently a placeholder, as the Resnet example just runs Resnet and logs to stdout. In any case, hopefully that worked. If it didn't... happy debugging!

### Monitoring

You can check in on your cluster using the handy k8s UI. At the command line, set up the proxy:

```
kubectl config use-context <cluster-id>
kubectl proxy
```

Then navigate to http://localhost:8001/ui in a browser window. You can authorize yourself by pointing the UI to your k8s config, which is likely `$HOME/.kube/config`. (Alternatively, you can use the token printed out in the config for your cluster when you run `kubectl config view`.)

Poke around and make sure your nodes, pods, and jobs are all error free. You can view the logs from containers directly through the UI, and those should be outputting information about Resnet training. If you are more command-line inclined, you can tail logs from your local machine:

```
kubectl logs --follow --timestamps=true <container-name>
```

### Troubleshooting

#### Specific issues

* Your cluster hangs at "Waiting for nodes to become stable": make sure that you have sufficient GPU quotas, and that you have requested GPUs in a region that has them available. You can check the GCE console where the instances and groups are spinning up; if you have GPU quota issues, there is likely a little yellow warning icon next to the minion instances. (Hover over the icon to read the warning.)

#### Debugging advice

* Note that you can use the GCP console to SSH into master or minion nodes directly as you debug. Check `/var/log/syslog` for errors.
* Permissions are a frequent cause of trouble. Make sure you can create instances from the command line using `gcloud`; check your quotas; explore the GCP console to make sure you're not missing something; etc.
* "Your local machine" here is a physical box outside of GCP; in theory, an instance inside of GCP could be used, but that complicates the permissions issues further. Sort those out before attempting the above.
* Part of the difficulty is that there are a million configuration details, and if any is not exactly correct, the pipeline can fail cryptically. When in doubt, remove bells and whistles-- can you run a vanilla kube-up.sh cluster? Can you run without GPUs, or without Ubuntu? Can you create a GPU instance from the command line? Etc.
