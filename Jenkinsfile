@Library(['github.com/indigo-dc/jenkins-pipeline-library@release/2.1.1']) _

// We choose to describe Docker image build directly in the Jenkinsfile instead of JePL2
// since this gives better control on the building process
// (e.g. which branches are allowed for docker image build)

def projectConfig

// function to remove built images
def docker_clean() {
    def dangling_images = sh(
	returnStdout: true,
	script: "docker images -f 'dangling=true' -q"
    )
    if (dangling_images) {
        sh(script: "docker rmi --force $dangling_images")
    }
}

pipeline {
    agent any

    environment {
        // Remove .git from the GIT_URL link and extract REPO_NAME from GIT_URL
        REPO_URL = "${env.GIT_URL.endsWith(".git") ? env.GIT_URL[0..-5] : env.GIT_URL}"
        REPO_NAME = "${REPO_URL.tokenize('/')[-1]}"
        AI4OS_REGISTRY_CREDENTIALS = credentials('AIOS-registry-credentials')
        APP_DOCKERFILE = "Dockerfile"
    }

    stages {
        stage("Variable initialization") {
            steps {
                script {
                    checkout scm
                    withFolderProperties{
                        env.DOCKER_REGISTRY = env.AI4OS_REGISTRY
                        env.DOCKER_REGISTRY_ORG = env.AI4OS_REGISTRY_REPOSITORY
                        env.DOCKER_REGISTRY_CREDENTIALS = env.AI4OS_REGISTRY_CREDENTIALS
                    }
                    // define tag based on branch
                    image_tag = "${env.BRANCH_NAME == 'main' ? 'latest' : env.BRANCH_NAME}"
                    // use REPO_NAME as Docker image name 
                    env.DOCKER_REPO = env.DOCKER_REGISTRY_ORG + "/" + env.REPO_NAME + ":" + image_tag
                    env.DOCKER_REPO = env.DOCKER_REPO.toLowerCase()
                    println ("[DEBUG] Config for the Docker image build: $env.DOCKER_REPO, push to $env.DOCKER_REGISTRY")
                }
            }
        }
        stage('Application testing') {
            steps {
                script {
                    projectConfig = pipelineConfig()
                    buildStages(projectConfig)
                }
            }
        }
        stage("Docker image building & delivery") {
            when {
                anyOf {
                    branch 'main'
                    branch 'release/*'
                    buildingTag()
                }
            }
            steps {
                script {
                    checkout scm
                    docker.withRegistry(env.DOCKER_REGISTRY, env.DOCKER_REGISTRY_CREDENTIALS){
                         def app_image = docker.build(env.DOCKER_REPO,
                                                      "--no-cache --force-rm --build-arg branch=${env.BRANCH_NAME} -f ${env.APP_DOCKERFILE} .")
                         app_image.push()
                    }
                }
            }
            post {
                failure {
                    docker_clean()
                }
            }
        }
    }
    post {
        // publish results and clean-up
        always {
            script {
                if (fileExists("flake8.log")) {
                    // publish results of the style analysis
                    recordIssues(tools: [flake8(pattern: 'flake8.log',
                                         name: 'PEP8 report',
                                         id: "flake8_pylint")])
                }
                if (fileExists("htmlcov/index.html")) {
                    // publish results of the coverage test
                    publishHTML([allowMissing: false,
                                 alwaysLinkToLastBuild: false,
                                 keepAll: true,
                                 reportDir: "htmlcov",
                                 reportFiles: 'index.html',
                                 reportName: 'Coverage report',
                                 reportTitles: ''])
                }
                if (fileExists("bandit/index.html")) {
                    // publish results of the security check
                    publishHTML([allowMissing: false, 
                                 alwaysLinkToLastBuild: false,
                                 keepAll: true,
                                 reportDir: "bandit",
                                 reportFiles: 'index.html',
                                 reportName: 'Bandit report',
                                 reportTitles: ''])
                }
            }
            // Clean after build
            cleanWs()
        }
    }
}

