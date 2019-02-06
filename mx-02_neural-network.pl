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
##  https://beta.mxnet.io/guide/crash-course/2-nn.html
    
##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  

use strict;
use warnings;
use AI::MXNet qw(mx);
use AI::MXNet::Gluon::NN qw(nn);

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "\n";
print "Create a neural network\n";
print "Create your neural network's first layer\n";
print "\n";

##  two output units
print "two output units:\n";
print 'my $layer = nn->Dense(2);'."\n";
my $layer = nn->Dense(2);
print $layer;
print "\n\n";

##  initialize the layer
print '$layer->initialize;'."\n";
$layer->initialize;

##  create shape and apply layer's input limit
print "create a (3,4) shape from random data\n";
print 'my $zz = nd->random->uniform(-1,1,[3,4]);';
my $zz = nd->random->uniform(-1,1,[3,4]);
#print $zz->aspdl;
print "\n"; print "\n";
print "apply the layer's input limit of 2\n";
print 'my $zlayer = $layer->($zz);';
my $zlayer = $layer->($zz);
print $zlayer->aspdl;
print "\n";

##  access the weight
print "access the weight\n";
print 'my $layerwd = $layer->weight->data;';
my $layerwd = $layer->weight->data;
print $layerwd->aspdl;
print "\n";

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "\n";
print "========================================\n";
print "\n";
print "Chain layers into a neural network\n";
print "\n";

##  NB:  the notes below were included in the Python original
##  https://beta.mxnet.io/guide/crash-course/2-nn.html
##  
##  Add a sequence of layers.
##  Similar to Dense, it is not necessary to specify the input channels
##  by the argument `in_channels`, which will be  automatically inferred
##  in the first forward pass. Also, we apply a relu activation on the
##  output. In addition, we can use a tuple to specify a  non-square
##  kernel size, such as `kernel_size=(2,4)`
## 
##  Add a sequence of layers.
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

print $net;
print "\n";
print '$net->initialize();';
$net->initialize();
print "\n";
print 'my $xx = nd->random->uniform(shape=>[4,1,28,28]);'."\n";
print 'my $yy = $net->($xx);'."\n";
my $xx = nd->random->uniform(shape=>[4,1,28,28]);
my $yy = $net->($xx);
print "\n";
print 'shape of $yy : '.'['. join(",", @{$yy->shape} ).']'."\n";
print "\n";
print 'shape of ${$net}[0] : '.'['. join(",", @{${$net}[0]->weight->data->shape }).']'."\n";
print 'shape of ${$net}[5] : '.'['. join(",", @{${$net}[5]->bias->data->shape }).']'."\n";

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "\n";
print "========================================\n";
print "\n";
print "Create a neural network flexibly\n";
print "\n";


{ package MixMLP;
  use strict;
  use warnings;
  use AI::MXNet::Gluon::Mouse;
  use AI::MXNet::Function::Parameters;
  extends 'AI::MXNet::Gluon::Block';
 
  sub BUILD {
      my $self = shift;
      $self->name_scope( 
	  sub {
	      $self->blk( nn->Sequential );
	      $self->blk->name_scope( 
		  sub { 
		      $self->blk->add( nn->Dense(3, activation=>'relu'));
		      $self->blk->add( nn->Dense(4, activation=>'relu'));
		  });
	      $self->dense( nn->Dense(5) );
	  });
  }
 
  method forward($xx) {
      my $yy = nd->relu( $self->blk->($xx) );
      return $self->dense->($yy);
  }
}

my $mixmlp = MixMLP->new();
print $mixmlp ;
print "\n";

##  define the initialization method
$mixmlp->initialize();

##  simulate batch of two and two inputs
my $ww = nd->random->uniform(shape=>[2,2]);

##  pass it to the network
$mixmlp->($ww);

print "\n";
print "weight of a particular layer:\n";
print '${$mixmlp->blk}[1]->weight->data->aspdl;';
print ${$mixmlp->blk}[1]->weight->data->aspdl ;
print "\n";


print "\n";
print "========================================\n";
print "\n";


##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  

##  SUBROUTINES

