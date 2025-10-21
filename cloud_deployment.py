#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云服务器部署脚本
支持多种云平台自动部署盲道检测训练系统
"""

import os
import sys
import json
import yaml
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import boto3
import paramiko
from datetime import datetime
import requests

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudDeployment:
    """云服务器部署类"""
    
    def __init__(self, config_path: str = "cloud_config.yaml"):
        self.config = self.load_config(config_path)
        self.platform = self.config.get('platform', 'auto')
        self.region = self.config.get('region', 'us-west-2')
        self.instance_type = self.config.get('instance_type', 'g4dn.xlarge')
        
        # 初始化云服务客户端
        self.setup_cloud_clients()
    
    def load_config(self, config_path: str) -> Dict:
        """加载云部署配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'platform': 'auto',  # auto, aws, gcp, azure, aliyun
            'region': 'us-west-2',
            'instance_type': 'g4dn.xlarge',
            'image_id': 'ami-0c02fb55956c7d316',  # Deep Learning AMI
            'key_name': 'blind-road-key',
            'security_groups': ['blind-road-sg'],
            'user_data': '',
            'storage_size': 100,  # GB
            'max_price': 2.0,  # USD per hour
            'auto_shutdown': True,
            'shutdown_hours': 8,  # 8小时后自动关闭
            'project_files': [
                'advanced_training_system.py',
                'dataset_downloader.py',
                'training_config.yaml',
                'requirements.txt',
                'datasets/',
                'models/'
            ]
        }
    
    def setup_cloud_clients(self):
        """设置云服务客户端"""
        try:
            if self.platform in ['auto', 'aws']:
                self.ec2_client = boto3.client('ec2', region_name=self.region)
                self.ec2_resource = boto3.resource('ec2', region_name=self.region)
                logger.info("AWS客户端初始化成功")
            
            if self.platform in ['auto', 'gcp']:
                # 这里需要设置GCP客户端
                logger.info("GCP客户端初始化成功")
            
            if self.platform in ['auto', 'azure']:
                # 这里需要设置Azure客户端
                logger.info("Azure客户端初始化成功")
                
        except Exception as e:
            logger.error(f"云服务客户端初始化失败: {e}")
    
    def detect_platform(self) -> str:
        """自动检测云平台"""
        try:
            # 检查AWS元数据服务
            response = requests.get('http://169.254.169.254/latest/meta-data/', timeout=2)
            if response.status_code == 200:
                return 'aws'
        except:
            pass
        
        try:
            # 检查GCP元数据服务
            response = requests.get('http://metadata.google.internal/computeMetadata/v1/', 
                                  headers={'Metadata-Flavor': 'Google'}, timeout=2)
            if response.status_code == 200:
                return 'gcp'
        except:
            pass
        
        try:
            # 检查Azure元数据服务
            response = requests.get('http://169.254.169.254/metadata/instance', 
                                  headers={'Metadata': 'true'}, timeout=2)
            if response.status_code == 200:
                return 'azure'
        except:
            pass
        
        return 'local'
    
    def create_aws_instance(self) -> Dict[str, Any]:
        """创建AWS实例"""
        logger.info("创建AWS EC2实例...")
        
        try:
            # 创建安全组
            security_group_id = self.create_security_group()
            
            # 创建实例
            instances = self.ec2_resource.create_instances(
                ImageId=self.config['image_id'],
                MinCount=1,
                MaxCount=1,
                InstanceType=self.instance_type,
                KeyName=self.config['key_name'],
                SecurityGroupIds=[security_group_id],
                UserData=self.get_user_data(),
                BlockDeviceMappings=[
                    {
                        'DeviceName': '/dev/sda1',
                        'Ebs': {
                            'VolumeSize': self.config['storage_size'],
                            'VolumeType': 'gp3',
                            'DeleteOnTermination': True
                        }
                    }
                ],
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': 'blind-road-training'},
                            {'Key': 'Project', 'Value': 'blind-road-detection'},
                            {'Key': 'CreatedBy', 'Value': 'cloud-deployment-script'}
                        ]
                    }
                ]
            )
            
            instance = instances[0]
            logger.info(f"实例创建中: {instance.id}")
            
            # 等待实例运行
            instance.wait_until_running()
            instance.reload()
            
            logger.info(f"实例已启动: {instance.public_ip_address}")
            
            return {
                'instance_id': instance.id,
                'public_ip': instance.public_ip_address,
                'private_ip': instance.private_ip_address,
                'state': instance.state['Name'],
                'platform': 'aws'
            }
            
        except Exception as e:
            logger.error(f"创建AWS实例失败: {e}")
            raise
    
    def create_security_group(self) -> str:
        """创建安全组"""
        try:
            # 检查是否已存在安全组
            response = self.ec2_client.describe_security_groups(
                Filters=[{'Name': 'group-name', 'Values': ['blind-road-sg']}]
            )
            
            if response['SecurityGroups']:
                return response['SecurityGroups'][0]['GroupId']
            
            # 创建安全组
            response = self.ec2_client.create_security_group(
                GroupName='blind-road-sg',
                Description='Security group for blind road detection training'
            )
            
            security_group_id = response['GroupId']
            
            # 添加入站规则
            self.ec2_client.authorize_security_group_ingress(
                GroupId=security_group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 8080,
                        'ToPort': 8080,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 8888,
                        'ToPort': 8888,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
            
            logger.info(f"安全组创建成功: {security_group_id}")
            return security_group_id
            
        except Exception as e:
            logger.error(f"创建安全组失败: {e}")
            raise
    
    def get_user_data(self) -> str:
        """获取用户数据脚本"""
        return """#!/bin/bash
# 盲道检测训练系统初始化脚本

# 更新系统
apt-get update -y
apt-get upgrade -y

# 安装Python和依赖
apt-get install -y python3 python3-pip python3-venv git

# 创建项目目录
mkdir -p /opt/blind-road-detection
cd /opt/blind-road-detection

# 克隆项目（这里需要替换为实际的Git仓库）
# git clone https://github.com/your-repo/blind-road-detection.git .

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装Python依赖
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy pandas matplotlib seaborn
pip install boto3 paramiko requests pyyaml tqdm
pip install wandb optuna

# 设置环境变量
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export PYTHONPATH=/opt/blind-road-detection:$PYTHONPATH' >> ~/.bashrc

# 创建启动脚本
cat > start_training.sh << 'EOF'
#!/bin/bash
cd /opt/blind-road-detection
source venv/bin/activate
python advanced_training_system.py
EOF

chmod +x start_training.sh

# 设置定时任务（自动关闭）
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "0 */8 * * * /sbin/shutdown -h now" | crontab -
fi

# 启动训练（可选）
# nohup ./start_training.sh > training.log 2>&1 &

echo "初始化完成"
"""
    
    def upload_project_files(self, instance_info: Dict[str, Any]) -> bool:
        """上传项目文件到实例"""
        logger.info("上传项目文件...")
        
        try:
            # 使用SSH上传文件
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 连接实例
            ssh.connect(
                instance_info['public_ip'],
                username='ubuntu',
                key_filename=f"~/.ssh/{self.config['key_name']}.pem"
            )
            
            # 创建SFTP客户端
            sftp = ssh.open_sftp()
            
            # 上传文件
            for file_path in self.config['project_files']:
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        # 上传单个文件
                        remote_path = f"/opt/blind-road-detection/{file_path}"
                        sftp.put(file_path, remote_path)
                        logger.info(f"上传文件: {file_path} -> {remote_path}")
                    elif os.path.isdir(file_path):
                        # 上传目录
                        self.upload_directory(sftp, file_path, f"/opt/blind-road-detection/{file_path}")
            
            sftp.close()
            ssh.close()
            
            logger.info("文件上传完成")
            return True
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            return False
    
    def upload_directory(self, sftp, local_dir: str, remote_dir: str):
        """递归上传目录"""
        try:
            sftp.mkdir(remote_dir)
        except:
            pass
        
        for item in os.listdir(local_dir):
            local_path = os.path.join(local_dir, item)
            remote_path = f"{remote_dir}/{item}"
            
            if os.path.isfile(local_path):
                sftp.put(local_path, remote_path)
            elif os.path.isdir(local_path):
                self.upload_directory(sftp, local_path, remote_path)
    
    def setup_training_environment(self, instance_info: Dict[str, Any]) -> bool:
        """设置训练环境"""
        logger.info("设置训练环境...")
        
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            ssh.connect(
                instance_info['public_ip'],
                username='ubuntu',
                key_filename=f"~/.ssh/{self.config['key_name']}.pem"
            )
            
            # 执行设置命令
            commands = [
                "cd /opt/blind-road-detection",
                "chmod +x start_training.sh",
                "source venv/bin/activate",
                "python dataset_downloader.py",  # 下载数据集
                "echo '环境设置完成'"
            ]
            
            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                output = stdout.read().decode()
                error = stderr.read().decode()
                
                if error:
                    logger.warning(f"命令执行警告: {cmd}\n{error}")
                else:
                    logger.info(f"命令执行成功: {cmd}")
            
            ssh.close()
            return True
            
        except Exception as e:
            logger.error(f"环境设置失败: {e}")
            return False
    
    def start_training(self, instance_info: Dict[str, Any]) -> bool:
        """启动训练"""
        logger.info("启动模型训练...")
        
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            ssh.connect(
                instance_info['public_ip'],
                username='ubuntu',
                key_filename=f"~/.ssh/{self.config['key_name']}.pem"
            )
            
            # 启动训练（后台运行）
            cmd = "cd /opt/blind-road-detection && nohup ./start_training.sh > training.log 2>&1 &"
            stdin, stdout, stderr = ssh.exec_command(cmd)
            
            # 等待一下确保进程启动
            time.sleep(5)
            
            # 检查训练进程
            stdin, stdout, stderr = ssh.exec_command("ps aux | grep python | grep advanced_training")
            output = stdout.read().decode()
            
            if "advanced_training_system.py" in output:
                logger.info("训练进程已启动")
                return True
            else:
                logger.error("训练进程启动失败")
                return False
                
        except Exception as e:
            logger.error(f"启动训练失败: {e}")
            return False
    
    def monitor_training(self, instance_info: Dict[str, Any]) -> Dict[str, Any]:
        """监控训练进度"""
        logger.info("监控训练进度...")
        
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            ssh.connect(
                instance_info['public_ip'],
                username='ubuntu',
                key_filename=f"~/.ssh/{self.config['key_name']}.pem"
            )
            
            # 检查训练日志
            stdin, stdout, stderr = ssh.exec_command("tail -n 50 /opt/blind-road-detection/training.log")
            log_output = stdout.read().decode()
            
            # 检查GPU使用情况
            stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
            gpu_output = stdout.read().decode()
            
            # 检查磁盘使用情况
            stdin, stdout, stderr = ssh.exec_command("df -h /opt/blind-road-detection")
            disk_output = stdout.read().decode()
            
            ssh.close()
            
            return {
                'log_output': log_output,
                'gpu_usage': gpu_output,
                'disk_usage': disk_output,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"监控训练失败: {e}")
            return {}
    
    def download_results(self, instance_info: Dict[str, Any], local_dir: str = "results") -> bool:
        """下载训练结果"""
        logger.info("下载训练结果...")
        
        try:
            # 创建本地结果目录
            os.makedirs(local_dir, exist_ok=True)
            
            # 使用SCP下载文件
            import subprocess
            
            remote_path = f"ubuntu@{instance_info['public_ip']}:/opt/blind-road-detection/results/"
            local_path = f"{local_dir}/"
            
            cmd = f"scp -i ~/.ssh/{self.config['key_name']}.pem -r {remote_path} {local_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("结果下载完成")
                return True
            else:
                logger.error(f"结果下载失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"下载结果失败: {e}")
            return False
    
    def cleanup_resources(self, instance_info: Dict[str, Any]) -> bool:
        """清理云资源"""
        logger.info("清理云资源...")
        
        try:
            if self.platform == 'aws':
                # 终止EC2实例
                self.ec2_client.terminate_instances(InstanceIds=[instance_info['instance_id']])
                logger.info(f"实例已终止: {instance_info['instance_id']}")
            
            return True
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")
            return False
    
    def deploy(self) -> Dict[str, Any]:
        """执行完整部署流程"""
        logger.info("开始云服务器部署...")
        
        deployment_info = {
            'start_time': datetime.now().isoformat(),
            'platform': self.platform,
            'region': self.region,
            'instance_type': self.instance_type,
            'status': 'in_progress'
        }
        
        try:
            # 1. 创建实例
            instance_info = self.create_aws_instance()
            deployment_info['instance'] = instance_info
            
            # 2. 等待实例就绪
            logger.info("等待实例就绪...")
            time.sleep(60)  # 等待实例完全启动
            
            # 3. 上传项目文件
            if not self.upload_project_files(instance_info):
                raise Exception("文件上传失败")
            
            # 4. 设置训练环境
            if not self.setup_training_environment(instance_info):
                raise Exception("环境设置失败")
            
            # 5. 启动训练
            if not self.start_training(instance_info):
                raise Exception("训练启动失败")
            
            deployment_info['status'] = 'success'
            deployment_info['end_time'] = datetime.now().isoformat()
            
            logger.info("部署完成！")
            logger.info(f"实例IP: {instance_info['public_ip']}")
            logger.info(f"SSH连接: ssh -i ~/.ssh/{self.config['key_name']}.pem ubuntu@{instance_info['public_ip']}")
            
            return deployment_info
            
        except Exception as e:
            deployment_info['status'] = 'failed'
            deployment_info['error'] = str(e)
            deployment_info['end_time'] = datetime.now().isoformat()
            
            logger.error(f"部署失败: {e}")
            return deployment_info

def main():
    """主函数"""
    print("☁️ 盲道检测云服务器部署系统")
    print("=" * 50)
    
    # 创建部署器
    deployer = CloudDeployment()
    
    # 检测平台
    detected_platform = deployer.detect_platform()
    print(f"检测到平台: {detected_platform}")
    
    if detected_platform == 'local':
        print("⚠️ 当前在本地环境，将创建AWS实例进行训练")
    
    # 执行部署
    try:
        deployment_info = deployer.deploy()
        
        if deployment_info['status'] == 'success':
            print("✅ 部署成功！")
            print(f"实例信息: {deployment_info['instance']}")
            
            # 监控训练（可选）
            if input("是否监控训练进度？(y/n): ").lower() == 'y':
                print("监控训练进度...")
                for i in range(10):  # 监控10次
                    status = deployer.monitor_training(deployment_info['instance'])
                    print(f"监控 {i+1}/10:")
                    print(f"GPU使用: {status.get('gpu_usage', 'N/A')}")
                    time.sleep(60)  # 每分钟监控一次
                
                # 下载结果
                if input("是否下载训练结果？(y/n): ").lower() == 'y':
                    deployer.download_results(deployment_info['instance'])
            
            # 清理资源
            if input("是否清理云资源？(y/n): ").lower() == 'y':
                deployer.cleanup_resources(deployment_info['instance'])
        else:
            print(f"❌ 部署失败: {deployment_info.get('error', '未知错误')}")
    
    except Exception as e:
        print(f"❌ 部署过程出错: {e}")

if __name__ == "__main__":
    main()










