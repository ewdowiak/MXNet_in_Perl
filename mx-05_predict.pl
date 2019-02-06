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

##  Perl replication of:
##  https://beta.mxnet.io/guide/crash-course/5-predict.html

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  

use strict;
use warnings;
use AI::MXNet qw(mx);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::AutoGrad qw(autograd);
use AI::MXNet::Base;

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "\n";
print "========================================\n";
print "define the model\n";
print "\n";

##  define the same model
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

##  load the parameters
my $param_file = './data/params/fashion-mnist.params';
$net->load_parameters($param_file);

##  print the model
print $net;
print "\n";

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "\n";
print "========================================\n";
print "load validation data and predict\n";
print "\n";

##  load the validation data
my $mnist_valid = gluon->data->vision->FashionMNIST( 
    root=>'./data/fashion-mnist', train=>0, transform=>\&transformer);
my $valid_data = gluon->data->DataLoader( 
    $mnist_valid, batch_size=>1, shuffle=>0);

##  define the same transformer function
sub transformer {
    my ($data, $label) = @_;
    $data = $data->nd::reshape([1,28,28]);
    $data = $data->astype('float32')/255;
    $data = ( $data - 0.31 ) / 0.31;
    return( $data , $label);
}

##  define the text labels
my @text_labels = ('t-shirt','trouser','pullover','dress','coat',
		   'sandal','shirt','sneaker','bag','ankle boot');

##  now take the first TEN Data-Labels
my @tendl = @{$valid_data}[0..9];

##  examine model predictions
my $topline ;
$topline .= '  PREDICTION :: CORRECT'."\n";
$topline .= '  ========== :: ======='."\n";
print $topline;
for my $i (0..$#tendl) {
    my $data  = ${$tendl[$i]}[0] ;
    my $label = ${$tendl[$i]}[1] ;

    my $ot = $net->($data)->argmax({axis=>1});
    my $pred = $text_labels[ PDL::sclr( $ot->aspdl )];
    my $true = $text_labels[ PDL::sclr( $label->aspdl )];

    my $otline;
    $otline .= sprintf("%12s",$pred) ." :: ";
    $otline .= sprintf("%-10s",$true)." ";
    $otline .= ( $pred eq $true ) ? ".." : "XX";

    print $otline ."\n";
}
print "\n";


##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
