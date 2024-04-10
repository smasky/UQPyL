import requests

def check_package_exists(package_name):
    response = requests.get(f'https://pypi.org/pypi/{package_name}/json')
    return response.status_code == 200

# 使用你的包名替换'your_package_name'
print(check_package_exists('UQPyL'))