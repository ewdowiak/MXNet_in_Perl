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
##  https://beta.mxnet.io/guide/crash-course/3-autograd.html

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  

use strict;
use warnings;
use AI::MXNet qw(mx);

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "\n";
print "Automatic differentiation with autograd\n";
print "Basic usage\n";
print "\n";

print "create an NDArray\n";
print 'my $xx = nd->array([[1, 2], [3, 4]]);'."\n";
my $xx = nd->array([[1, 2], [3, 4]]);
print $xx->aspdl;
print "\n";

print "invoke attach_grad\n";
print '$xx->attach_grad;'."\n";
$xx->attach_grad;
print "\n";

print "record the definition\n";
my $yy;
mx->autograd->record(sub {
    $yy = 2 * $xx * $xx ;
});
#print $yy->aspdl;
print "\n";

print "invoke back propagation\n";
print '$yy->backward;'."\n";
$yy->backward;
#print "\n";

print "and check the output\n";
print 'print $xx->grad->aspdl;'."\n";
print $xx->grad->aspdl;
print "\n";
    
##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "\n";
print "========================================\n";
print "Using control flows\n";
print "\n";

print "define a function\n";

sub pdlmnorm {
    require PDL::LinearAlgebra;
    my $invec = $_[0] ;
    my $mnorm = PDL::LinearAlgebra::mnorm( $invec->aspdl );
    $mnorm = PDL::sclr( $mnorm );
    return $mnorm;
}

sub ff { 
    my $aa = $_[0];
    my $bb = $aa * 2; 

    while ( pdlmnorm($bb) < 1000) {
	$bb = $bb *2;
    }
    
    my $cc;
    if ( $bb->sum >= 0 ) {
	$cc = $bb->slice('0');
    } else {
	$cc = $bb->slice('1');
    }
    return nd->array($cc);
}


print "record trace and feed in random values\n";
my $aa = nd->random->uniform(shape=>2);
$aa->attach_grad;

my $cc;
mx->autograd->record(sub {
    $cc = ff($aa) ;
});
print "invoke back propagation\n";
$cc->backward;

print "\n";
print "the argument to the function\n";
print $aa->aspdl;
print "\n\n";
print "the value of the function\n";
print $cc->aspdl;
print "\n\n";

print "the gradient\n";
print $aa->grad->aspdl;
print "\n";
my $alt = $cc/$aa;
print $alt->aspdl;
print "\n";
print "\n";

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  

##  SUBROUTINES

