# -*- mode: ruby -*-
# # vi: set ft=ruby :

#$script = <<-SCRIPT
#sudo apt install gcc g++
#SCRIPT


Vagrant.configure("2") do |config|
	config.vm.define "box" do |box|
				box.vm.box = "ubuntu/bionic64"
				box.vm.hostname = "bpnn"
				box.vm.provider "virtualbox" do |virtualbox|
			virtualbox.name="bpnn"
		end
	end
	#config.vm.provision "shell", inline: $script
end
