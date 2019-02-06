#!/usr/bin/env perl 

##  Copyright 2019 Eryk Wdowiak
##  
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##  
##      http://www.apache.org/licenses/LICENSE-2.0
##  
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  

##  NOTE:  This script replicates the "Train the Neural Network" page of the 
##  MXNet Crash Course [1], using large amounts of code from Sergey Kolychev's
##  "mnist.pl" example [2]. In particular, the "transformer," "test" and "train" 
##  subroutines were originally written by S. Kolychev.  Everything else is just 
##  an adaptation for the Fashion-MNIST dataset, which is used in the tutorial.
##  
##  References:
##  [1] https://beta.mxnet.io/guide/crash-course/4-train.html
##  [2] https://metacpan.org/source/SKOLYCHEV/AI-MXNet-1.33/examples/gluon/mnist.pl

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  

use strict;
use warnings;
use AI::MXNet qw(mx);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::AutoGrad qw(autograd);
use AI::MXNet::Base;
use Getopt::Long;

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

GetOptions(
    'lr=f'          => \( my $lr          =  0.1 ),
    'momentum=f'    => \( my $momentum    =  0.0 ),
    'cuda=i'        => \( my $cuda        =  0   ),
    'batch-size=i'  => \( my $batch_size  =  256 ),
    'epochs=i'      => \( my $epochs      =   10 ),
    'help'          => \( my $help        =   '' ),
    ) or GetHelp(1);
GetHelp(0) if ($help ne '');

my $param_file = './data/params/fashion-mnist.params';

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

##  get the data
my $mnist_train = gluon->data->vision->FashionMNIST( 
    root=>'./data/fashion-mnist', train=>1, transform=>\&transformer);
my $mnist_valid = gluon->data->vision->FashionMNIST(
    root=>'./data/fashion-mnist', train=>0, transform=>\&transformer);

##  load the data
my $train_data = gluon->data->DataLoader( $mnist_train,
    batch_size=>$batch_size, shuffle=>1);
my $valid_data = gluon->data->DataLoader( $mnist_valid,
    batch_size=>$batch_size, shuffle=>0);

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

##  define the model
my $net = nn->Sequential();
$net->name_scope(sub {
    $net->add(nn->Conv2D(channels=>6, kernel_size=>5, activation=>'relu'));
    $net->add(nn->MaxPool2D(pool_size=>2, strides=>2));
    $net->add(nn->Conv2D(channels=>16, kernel_size=>3, activation=>'relu'));
    $net->add(nn->MaxPool2D(pool_size=>2, strides=>2));
    $net->add(nn->Flatten()),
    $net->add(nn->Dense(120, activation=>"relu"));
    $net->add(nn->Dense(84, activation=>"relu"));
    $net->add(nn->Dense(10));
		 });
print "\n";
print $net;
print "\n";

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

##  train the model
my $ctx = $cuda ? mx->gpu(0) : mx->cpu();
train($epochs, $ctx);


##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

##  SUBROUTINES
##  ===========

##  get help
sub GetHelp {
    my $excode = $_[0];
    my $msg = "\n";
    $msg .= $0 ."\n";
    $msg .= "\t". '--lr=0.1'."\n";
    $msg .= "\t". '--momentum=0.0'."\n";
    $msg .= "\t". '--cuda=0'."\n";
    $msg .= "\t". '--batch-size=256'."\n";
    $msg .= "\t". '--epochs=10'."\n";
    $msg .= "\t". '--help'."\n";
    print $msg ."\n";
    exit $excode;
}


##  transform the input data
sub transformer {
    my ($data, $label) = @_;
    $data = $data->reshape([1,28,28]);
    $data = $data->astype('float32')/255;
    $data = ( $data - 0.31 ) / 0.31;
    return( $data , $label);
}

##  test model accuracy
sub test {

    my $ctx = shift;
    my $metric = mx->metric->Accuracy();

    while( defined( my $d = <$valid_data>)) {

        my ($data, $label) = @$d;
        $data  = $data->as_in_context($ctx);
        $label = $label->as_in_context($ctx);

	##  run data through the network
        my $output = $net->($data);
        $metric->update([$label], [$output]);
    }
    return $metric->get;
}

##  train the model
sub train {

    my ($epochs, $ctx) = @_;

    ##  Collect all parameters from net and its children, then initialize them.
    $net->initialize(mx->init->Xavier(), ctx=>$ctx); 

    ##  Trainer is for updating parameters with gradient.
    my $trainer = gluon->Trainer( 
	$net->collect_params(), 'sgd', 
	{learning_rate => $lr, momentum => $momentum});

    ##  metric and loss
    my $metric = mx->metric->Accuracy();
    my $loss = gluon->loss->SoftmaxCrossEntropyLoss();

    for my $epoch (0..$epochs-1) {

	##  set scalars to hold time and mean loss
	my $time = time();
	my $llm;

        ##  reset data iterator and metric at begining of epoch
        $metric->reset();
        enumerate(sub {
            my ($i, $d) = @_;
            my ($data, $label) = @$d;
	    $data  = $data->as_in_context($ctx);
	    $label = $label->as_in_context($ctx);

            ##  Start recording computation graph with record() section.
            ##  Recorded graphs can then be differentiated with backward.
            my $output;
	    my $LL;
            autograd->record( sub{
                $output = $net->($data);
                $LL = $loss->($output, $label);});
	    $LL->backward;

	    ##  capture the mean loss
	    $llm = PDL::sclr($LL->mean->aspdl);

            ##  take a gradient step with batch_size equal to data.shape[0]
            $trainer->step($data->shape->[0]);

            ##  update metric 
            $metric->update([$label], [$output]);

        }, \@{ $train_data });
	##  end of epoch
	##  the argument to the subroutine is: \@{$train_data}
	
	##  get training accuracy
        my ($trn_name, $trn_acc) = $metric->get();

	##  get validation accuracy
        my ($val_name, $val_acc) = test($ctx);

	##  capture time interval
	my $nowtime  = time();
	my $interval = $nowtime - $time;
	
	##  print the result 
	my $ottxt;
	$ottxt .= "Epoch ". $epoch .":  ";
	$ottxt .= "loss "      . sprintf( "%.3f", $llm) .",  ";
	$ottxt .= "train acc " . sprintf( "%.3f", $trn_acc) .",  ";
	$ottxt .= "test acc "  . sprintf( "%.3f", $val_acc) ."  ";
	$ottxt .= "in " . $interval  ." secs";
	print $ottxt ."\n";
    }
    
    ##  done training, so save parameters
    print "done training.  saving parameters."."\n";
    $net->save_parameters($param_file);
    print "\n";
}
