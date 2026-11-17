pipeline {
	agent any
	stages {
		stage('Build image')
		{
			steps{
				sh 'docker build -t fraud-detection:${BUILD_NUMBER} .'
			}
		}
		stage('Test'){
			steps{
				sh 'docker run --rm fraud-detection:${BUILD_NUMBER} python -m pytest -v'
			}
		}
	}
	post{
		always {
			sh 'docker rmi fraud-detection:${BUILD_NUMBER} || true'
		}
	}
}
