%MIT License
%
%Copyright (c) 2017 Ray Richardson
%
%Permission is hereby granted, free of charge, to any person obtaining a copy
%of this software and associated documentation files (the "Software"), to deal
%in the Software without restriction, including without limitation the rights
%to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
%copies of the Software, and to permit persons to whom the Software is
%furnished to do so, subject to the following conditions:
%
%The above copyright notice and this permission notice shall be included in all
%copies or substantial portions of the Software.
%
%THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
%IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
%FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
%AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
%LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
%OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
%SOFTWARE.

/**
  This module implements the Particle Swarm Optimization in General Terms
  The PSO can be used to heuristically find an optimal solution to many
  optimization problems. Particles are represented by a Floating Point Vector.
  @author Ray Richardson
  @license MIT
*/

:- module(pso, [pso_init/5, pso/3, apply_pso/4]).

:- thread_local omega/1, phi/2.

get_omega(Omega) :-
	(omega(Omega) ->
	 true
	;
	 Omega = 0.5).

get_phi(Phi_p, Phi_g) :-
	(phi(Phi_p, Phi_g) ->
	 true
	;
	 Phi_p = 0.5,
	 Phi_g = 0.5).

float_vec(Vector) :-
	(maplist(number, Vector) ->
	 true
	;
	 type_error(float_vec, Vector)).

:- meta_predicate pso_init(+, +, +, 2, -).
pso_init(ParticleCount, Blo, Bhi,  Objective, Swarm) :-
	float_vec(Blo),
	float_vec(Bhi), 
	length(Blo, LoCount),
	length(Bhi, HiCount),
	(LoCount =\= HiCount ->
	 domain_error('vectors not the same size', Blo-Bhi)
	;
	 true), 

	/**
	  Initialize Particles
	*/
	length(Particles, ParticleCount), 
	maplist(init_particle(Blo, Bhi), Particles),
	maplist(Objective, Particles, Values),
	
	/**
	  Initialize Velocity to Zero
	*/
	
	length(Velocity, LoCount),
	maplist(=(0.0), Velocity),
	
	/**
	  Build the Swarm List
	*/
	
	maplist(swarm_list(Velocity), Particles, Values, SwarmList),
%SwarmList = [First | Rest],
%foldl(swarm_leader, Rest, First, Leader),
	list_min(SwarmList, Leader), 
	Swarm = swarm(SwarmList, Leader, Blo, Bhi).


init_particle(Blo, Bhi, Particle) :-
	maplist(init_particle_dimension, Blo, Bhi, Particle).

init_particle_dimension(Lo, Hi, Value) :-
	random(Rand),
	Value is (Rand * (Hi - Lo)) + Lo.
	
	
get_swarm_pos(Particles, SwarmPos) :-
	list_min(Particles, SwarmPos).

swarm_list(Velocity, Particle, Value, particle(Value, Particle, Value, Particle, Velocity)).

:- meta_predicate pso(2, +, -).
pso(Objective, Swarm, NewSwarm) :-

	Swarm = swarm(SwarmList1, Leader1, Blo, Bhi),
	NewSwarm = swarm(SwarmList2, Leader2, Blo, Bhi),

	maplist(update_velocity(Leader1), SwarmList1, SwarmList_v),
	maplist(update_position, SwarmList_v, SwarmList_p),
	maplist(evaluate(Objective), SwarmList_p, SwarmList2),
	list_min(SwarmList2, Leader2).

vec_plus(V1, V2, V3) :-
	(maplist(val_plus, V1, V2, V3) ->
	 true
	;
	 domain_error('Vectors not the same Size', V1-V2)).

val_plus(A, B, C) :-
	C is A + B.

update_velocity(particle(_BestValue, Lead, _Val, _Particle, _Velocity), 
		particle(BestVal, Best, Value, Particle, Velocity),
		particle(BestVal, Best, Value, Particle, NewVelocity)) :-

	get_omega(Omega),
	get_phi(Phi_p, Phi_g),

	maplist(update_velocity_dim(Omega, Phi_p, Phi_g), Lead, Particle, Best, Velocity,
		NewVelocity).

update_velocity_dim(Omega, Phi_p, Phi_g, LeadDim, PartDim, BestDim, VelDim, NewVelDim) :-
	Rp is random_float,
	Rg is random_float,

	NewVelDim is Omega * VelDim + Phi_p * Rp * (BestDim - PartDim) + Phi_g * Rg * (LeadDim - PartDim).

update_position(particle(BestVal, Best, Value, Particle, Velocity),
		particle(BestVal, Best, Value, NewParticle, Velocity)) :-

	vec_plus(Particle, Velocity, NewParticle).

:- meta_predicate evaluate(2, +, -).

evaluate(Objective, particle(BestValue, Best, _Value, Particle, Velocity),
	 particle(NewBestValue, NewBest, Value, Particle, Velocity)) :-

	call(Objective, Particle, Value),
	(Value < BestValue ->
	 NewBestValue = Value,
	 NewBest = Particle
	;
	 NewBestValue = BestValue,
	 NewBest = Best).


list_min([H | T], Min) :-
	foldl(min_val, T, H, Min).

min_val(Val, Last, Next) :-
	(Val @< Last ->
	 Next = Val
	;
	 Next = Last).


:- meta_predicate apply_pso(2, +, +).

apply_pso(Objective, Count, Swarm, FinalSwarm) :-
	(Count > 0 ->
	 pso(Objective, Swarm, NewSwarm),
	 NewSwarm = swarm(_List, Leader, _, _),
	 Leader = particle(Value, _, _, _, _),
	 writeln(Value), 
	 Next is Count - 1, !,
	 apply_pso(Objective, Next, NewSwarm, FinalSwarm)
	;
	 FinalSwarm = Swarm).


:- meta_predicate maplist(5, ?, ?, ?, ?, ?).
:- meta_predicate maplist_(?, ?, ?, ?, ?, 5).

maplist_([], [], [], [], [], _Objective).
maplist_([H1 | T1], [H2 | T2], [H3 | T3], [H4 | T4], [H5 | T5], Objective) :-
	call(Objective, H1, H2, H3, H4, H5),
	!,
	maplist_(T1, T2, T3, T4, T5, Objective).

maplist(Objective, L1, L2, L3, L4, L5) :-
	maplist_(L1, L2, L3, L4, L5, Objective).

	