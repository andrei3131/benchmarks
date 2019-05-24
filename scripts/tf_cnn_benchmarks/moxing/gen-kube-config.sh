#!/bin/sh
set -e

KUBE_SA_CONFIG=/var/run/secrets/kubernetes.io/serviceaccount
if [ -d $KUBE_SA_CONFIG ]; then
    NAMESPACE=$(cat $KUBE_SA_CONFIG/namespace)
    TOKEN=$(cat $KUBE_SA_CONFIG/token)
fi

kubectl config set-cluster this --server https://kubernetes/ --certificate-authority=$KUBE_SA_CONFIG/ca.crt
kubectl config set-credentials me --token "$TOKEN"
kubectl config set-context me@this --cluster=this --user=me --namespace "$NAMESPACE"
kubectl config use me@this
